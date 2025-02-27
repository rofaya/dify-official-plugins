import hashlib
import json
import time
from typing import Any, Generator, Mapping, Optional

from dify_plugin import Endpoint
from loguru import logger
from pydantic import BaseModel
from werkzeug import Request, Response


class ConversationInfo(BaseModel):
    """存储对话信息的模型"""

    conversation_id: str
    created_at: float
    user_id: str


class OpenaiCompatible(Endpoint):
    @logger.catch
    def _invoke(self, r: Request, values: Mapping, settings: Mapping) -> Response:
        """
        Invokes the endpoint with the given request.
        """
        app_id: str = settings.get("app_id", {}).get("app_id", "")
        if not app_id:
            raise ValueError("App ID is required")
        if not isinstance(app_id, str):
            raise ValueError("App ID must be a string")

        memory_mode: str = settings.get("memory_mode", "last_user_message")
        try:
            data = r.get_json()
            logger.debug(f"收到请求: {json.dumps(data, indent=2, ensure_ascii=False)}")

            messages = data.get("messages", [])
            stream = data.get("stream", False)
            user = data.get("user", "")

            logger.info(f"收到请求: 用户={user}, 消息数量={len(messages)}")

            # 记录关键消息内容，便于调试
            if messages:
                first_msg = messages[0]
                last_msg = messages[-1]
                logger.debug(
                    f"首条消息: role={first_msg.get('role')}, content前20字符={first_msg.get('content', '')[:20]}"
                )
                logger.debug(
                    f"末条消息: role={last_msg.get('role')}, content前20字符={last_msg.get('content', '')[:20]}"
                )

            conversation_id, query = self._get_memory(memory_mode, messages, user)
            logger.info(
                f"获取记忆结果: conversation_id={conversation_id}, query前20字符={query[:20] if query else ''}"
            )

            if stream:

                def generator():
                    logger.debug(
                        f"开始流式响应: app_id={app_id}, conversation_id={conversation_id}"
                    )
                    response = self.session.app.chat.invoke(
                        app_id=app_id,
                        inputs={"messages": json.dumps(messages), "user": user},
                        query=query,
                        response_mode="streaming",
                        conversation_id=conversation_id,
                    )

                    # 如果是新会话（conversation_id为空），从首个响应中获取conversation_id并保存
                    if not conversation_id:
                        try:
                            first_chunk = next(response, None)
                            if first_chunk:
                                new_conversation_id = first_chunk.get("conversation_id", "")
                                logger.info(
                                    f"新会话首个响应块: conversation_id={new_conversation_id}"
                                )
                                if new_conversation_id:
                                    self._backtracking_conversation_id(
                                        messages, user, new_conversation_id
                                    )
                                yield from self._handle_chat_stream_message(
                                    app_id, [first_chunk, *response]
                                )
                            else:
                                logger.warning("未获取到首个响应块")
                                yield from self._handle_chat_stream_message(app_id, response)
                        except Exception as e:
                            logger.error(f"处理首个响应块时出错: {str(e)}")
                            yield from self._handle_chat_stream_message(app_id, response)
                    else:
                        yield from self._handle_chat_stream_message(app_id, response)

                return Response(
                    generator(),
                    status=200,
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Transfer-Encoding": "chunked"},
                )
            else:
                logger.debug(f"开始阻塞响应: app_id={app_id}, conversation_id={conversation_id}")
                response = self.session.app.chat.invoke(
                    app_id=app_id,
                    inputs={"messages": json.dumps(messages), "user": user},
                    query=query,
                    response_mode="blocking",
                    conversation_id=conversation_id,
                )

                # 如果是新会话，保存conversation_id
                if not conversation_id and response.get("conversation_id"):
                    new_conversation_id = response.get("conversation_id")
                    logger.info(f"新会话阻塞响应: conversation_id={new_conversation_id}")
                    self._backtracking_conversation_id(messages, user, new_conversation_id)

                return Response(
                    self._handle_chat_blocking_message(app_id, response),
                    status=200,
                    content_type="text/html",
                )
        except ValueError as e:
            logger.error(f"请求值错误: {str(e)}")
            return Response(f"Error: {e}", status=400, content_type="text/plain")
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            return Response(f"Error: {e}", status=500, content_type="text/plain")

    def _generate_conversation_key(self, messages: list[dict[str, Any]], user: str) -> str:
        """
        生成会话的唯一键
        使用消息内容、用户ID和时间戳的组合确保唯一性
        """
        # 提取前两轮对话作为特征（如果存在）
        conversation_feature = []
        system_message = None

        # 首先查找系统消息
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
                break

        if system_message:
            conversation_feature.append(f"system:{system_message}")

        # 提取第一条用户消息作为主要特征
        first_user_message = None
        for msg in messages:
            if msg.get("role") == "user":
                first_user_message = msg.get("content", "")
                break

        if first_user_message:
            conversation_feature.append(f"first_user:{first_user_message}")

        # 如果无法提取足够的特征，使用全部消息
        if len(conversation_feature) < 1:
            for i, msg in enumerate(messages):
                if i > 3:  # 限制消息数量，避免键过长
                    break
                conversation_feature.append(f"{msg.get('role')}:{msg.get('content', '')[:50]}")

        # 组合特征并添加用户ID
        feature_str = "||".join(conversation_feature) + f"||user:{user}"

        # 生成哈希值作为键
        key = hashlib.sha256(feature_str.encode()).hexdigest()
        logger.debug(f"生成会话键: key={key}, 基于特征数={len(conversation_feature)}")
        return key

    def _safe_storage_get(self, key: str) -> Optional[bytes]:
        """安全获取存储数据，处理所有可能的异常"""
        try:
            logger.info(f"尝试从存储中获取: key={key}")
            value = self.session.storage.get(key)
            logger.info(f"存储获取成功: key={key}, value长度={len(value) if value else 0}")
            return value
        except KeyError:
            logger.info(f"存储中不存在键: key={key}")
            return None
        except Exception as e:
            logger.error(f"从存储获取时发生异常: key={key}, error={str(e)}")
            return None

    def _safe_storage_set(self, key: str, value: bytes) -> bool:
        """安全设置存储数据，处理所有可能的异常"""
        try:
            logger.info(f"尝试写入存储: key={key}, value长度={len(value)}")
            self.session.storage.set(key, value)
            logger.info(f"存储写入成功: key={key}")
            return True
        except Exception as e:
            logger.error(f"写入存储时发生异常: key={key}, error={str(e)}")
            return False

    def _backtracking_conversation_id(
        self, messages: list[dict[str, Any]], user: str, conversation_id: str
    ) -> None:
        """
        存储消息和conversation_id之间的映射关系

        Args:
            messages: OpenAI格式的消息列表
            user: 用户标识
            conversation_id: Dify的会话ID
        """
        if not conversation_id:
            logger.warning("无法记录空conversation_id")
            return

        # 生成会话的唯一键
        conversation_key = self._generate_conversation_key(messages, user)

        # 创建会话信息对象
        conversation_info = ConversationInfo(
            conversation_id=conversation_id, created_at=time.time(), user_id=user
        )

        # 序列化数据
        try:
            data = conversation_info.model_dump_json().encode("utf-8")
            # 将会话信息存储到持久化存储中
            success = self._safe_storage_set(conversation_key, data)
            if success:
                logger.info(
                    f"成功存储会话映射: key={conversation_key}, conversation_id={conversation_id}"
                )
            else:
                logger.warning(f"存储会话映射失败: key={conversation_key}")
        except Exception as e:
            logger.error(f"序列化或存储会话信息时出错: {str(e)}")

    def _get_memory(
        self, memory_mode: str, messages: list[dict[str, Any]], user: str
    ) -> tuple[str, str]:
        """
        Get the memory from the messages

        returns:
            - conversation_id: str
            - query: str
        """
        if memory_mode == "last_user_message":
            # 获取最后一条用户消息作为查询
            user_message = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    user_message = message.get("content")
                    break

            if not user_message:
                logger.error("未找到用户消息")
                raise ValueError("No user message found")

            logger.info(f"提取的用户查询: {user_message[:30]}...")

            # 判断是否为多轮对话
            has_assistant_message = False
            for message in messages:
                if message.get("role") == "assistant":
                    has_assistant_message = True
                    break

            logger.info(f"是否为多轮对话: {has_assistant_message}")

            # 如果有assistant消息，表示这是多轮对话，尝试查找已存储的conversation_id
            if has_assistant_message:
                conversation_key = self._generate_conversation_key(messages, user)

                # 从存储中安全获取数据
                stored_data = self._safe_storage_get(conversation_key)

                if stored_data:
                    try:
                        # 尝试使用model_validate_json方法解析JSON
                        try:
                            conversation_info = ConversationInfo.model_validate_json(
                                stored_data.decode("utf-8")
                            )
                        except Exception:
                            # 如果上面的方法失败，尝试使用parse_raw方法
                            conversation_info = ConversationInfo.parse_raw(
                                stored_data.decode("utf-8")
                            )

                        logger.info(
                            f"找到已存储的conversation_id: {conversation_info.conversation_id}"
                        )
                        return conversation_info.conversation_id, user_message
                    except Exception as e:
                        logger.error(f"解析存储数据失败: {str(e)}")
                        # 如果解析失败，当作新会话处理
                else:
                    logger.info("未找到已存储的会话信息")

            # 如果是新会话或未找到存储的conversation_id，返回空字符串
            logger.info("返回空conversation_id，作为新会话处理")
            return "", user_message
        else:
            logger.error(f"不支持的memory_mode: {memory_mode}")
            raise ValueError(
                f"Invalid memory mode: {memory_mode}, only support last_user_message for now"
            )

    def _handle_chat_stream_message(
        self, app_id: str, generator: Generator[dict[str, Any], None, None]
    ) -> Generator[str, None, None]:
        """
        Handle the chat stream
        """
        message_id = ""
        for data in generator:
            if data.get("event") == "agent_message" or data.get("event") == "message":
                message = {
                    "id": "chatcmpl-" + data.get("message_id", "none"),
                    "object": "chat.completion.chunk",
                    "created": int(data.get("created", 0)),
                    "model": "gpt-3.5-turbo",
                    "system_fingerprint": "difyai",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": data.get("answer", "")},
                            "finish_reason": None,
                        }
                    ],
                }
                message_id = message.get("id", "none")
                yield f"data: {json.dumps(message)}\n\n"
            elif data.get("event") == "message_end":
                message = {
                    "id": "chatcmpl-" + data.get("message_id", "none"),
                    "object": "chat.completion.chunk",
                    "created": int(data.get("created", 0)),
                    "model": "gpt-3.5-turbo",
                    "system_fingerprint": "difyai",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "completion_tokens": data.get("metadata", {})
                        .get("usage", {})
                        .get("completion_tokens", 0),
                        "prompt_tokens": data.get("metadata", {})
                        .get("usage", {})
                        .get("prompt_tokens", 0),
                        "total_tokens": data.get("metadata", {})
                        .get("usage", {})
                        .get("total_tokens", 0),
                    },
                }
                yield f"data: {json.dumps(message)}\n\n"
            elif data.get("event") == "message_file":
                url = data.get("url", "")
                message = {
                    "id": "chatcmpl-" + message_id,
                    "object": "chat.completion.chunk",
                    "created": int(data.get("created", 0)),
                    "model": "gpt-3.5-turbo",
                    "system_fingerprint": "difyai",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": f"[{data.get('id', 'none')}]({url})",
                            },
                        }
                    ],
                }
                yield f"data: {json.dumps(message)}\n\n"

        yield "data: [DONE]\n\n"

    def _handle_chat_blocking_message(self, app_id: str, response: dict[str, Any]) -> str:
        """
        Handle the chat blocking message
        """
        message = {
            "id": "chatcmpl-" + response.get("id", "none"),
            "object": "chat.completion",
            "created": int(response.get("created", 0)),
            "model": "gpt-3.5-turbo",
            "system_fingerprint": "difyai",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.get("answer", "")},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": response.get("metadata", {})
                .get("usage", {})
                .get("completion_tokens", 0),
                "prompt_tokens": response.get("metadata", {})
                .get("usage", {})
                .get("prompt_tokens", 0),
            },
        }

        return json.dumps(message)
