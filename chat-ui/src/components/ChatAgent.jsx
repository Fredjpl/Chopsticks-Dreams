import { useState } from "react";
import axios from "axios";

export default function ChatAgent() {
  const [input, setInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newHistory = [...chatHistory, { role: "user", content: input }];
    setChatHistory(newHistory);
    setInput("");
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/chat", {
        question: input,
        chat_history: newHistory,
      });
      setChatHistory([
        ...newHistory,
        { role: "ai", content: res.data.answer || "（无返回）" },
      ]);
    } catch (err) {
      setChatHistory([
        ...newHistory,
        { role: "ai", content: "❌ 出错了，请稍后再试。" },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-2xl mx-auto p-4 space-y-4">
      <h1 className="text-2xl font-bold text-center">🧠 AI Agent 聊天界面</h1>
      <div className="space-y-2">
        {chatHistory.map((msg, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg ${
              msg.role === "user" ? "bg-blue-100 text-right" : "bg-gray-100"
            }`}
          >
            <p>
              <strong>{msg.role === "user" ? "你" : "AI"}：</strong>
              {msg.content}
            </p>
          </div>
        ))}
        {loading && <p className="text-gray-500">AI 正在思考中...</p>}
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 border rounded p-2"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="请输入问题..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          onClick={sendMessage}
        >
          发送
        </button>
      </div>
    </div>
  );
}
