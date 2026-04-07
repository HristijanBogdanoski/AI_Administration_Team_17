import { useState, useRef, useEffect } from "react";

const brandBlue = '#2e3e77';
const lightBlue = '#F0F4F8';

// Секое прашање има cat (категорија) — исто како faqData во FAQ.jsx
const QUICK_QUESTIONS = [
  { cat: "Документи", q: "Kako да добијам пасош?" },
  { cat: "Даноци",    q: "Рок за даночна пријава" },
  { cat: "Документи", q: "Документи за лична карта" },
  { cat: "Документи", q: "Возачка дозвола – барање" },
  { cat: "Социјала",  q: "Социјална помош услови" },
  { cat: "Локации",   q: "Закажи термин во МВР" },
];

const TOPICS = ["Сите", "Документи", "Даноци", "Социјала", "Локации", "Плаќање"];

const QA_MAP = {
  "Kako да добијам пасош?": {
    answer: "За добивање пасош потребно е:\n• Лична карта или извод од матична книга на родени\n• Две фотографии (35x45 мм)\n• Уплатница за административна такса\n• Барање поднесено во МВР\n\nПасошот е достапен за 5 работни дена.\n\nДали имате конкретно прашање за пасошот?",
  },
  "Рок за даночна пријава": {
    answer: "За даночна пријава:\n• Рокот за поднесување е 15 март секоја година\n• Пријавата може да се поднесе онлајн преку УЈП порталот\n• Тел: +389 2 3299 000\n\nДали имате конкретно прашање за вашата даночна пријава?",
  },
  "Документи за лична карта": {
    answer: "За лична карта потребно е:\n• Извод од матична книга на родени\n• Уверение за државјанство\n• Уплатница за административна такса\n• Барање поднесено лично во МВР\n\nЛичната карта се издава во рок од 15 работни дена.",
  },
  "Возачка дозвола – барање": {
    answer: "За возачка дозвола потребно е:\n• Важечка лична карта или пасош\n• Лекарско уверение (не постаро од 6 месеци)\n• Уплатница за такса\n• Положен возачки испит\n\nВозачката дозвола се обновува на секои 10 години.",
  },
  "Социјална помош услови": {
    answer: "Услови за социјална помош:\n• Приход под утврдениот минимум\n• Пријавено живеалиште во РСМ\n• Барање поднесено во Центарот за социјална работа\n\nТел: +389 2 3230 401",
  },
  "Закажи термин во МВР": {
    answer: "За закажување термин во МВР:\n• Посетете го порталот: uslugi.gov.mk\n• Изберете ја услугата која ви е потребна\n• Одберете датум и час\n• Добивате потврда на е-маил\n\nТел за информации: +389 2 3117 222",
  },
};

function formatAnswer(text) {
  return text.split("\n").map((line, i) => (
    <span key={i} style={{ display: "block", marginBottom: line === "" ? 8 : 2 }}>
      {line}
    </span>
  ));
}

function now() {
  return new Date().toLocaleTimeString("mk-MK", { hour: "2-digit", minute: "2-digit" });
}

export default function EVladaChatbot() {
  const [messages, setMessages] = useState([{
    id: 1, from: "bot", time: now(),
    text: "Добредојдовте! Јас сум е-Влада АИ Асистент. Изберете прашање или напишете го вашето прашање на македонски.",
  }]);
  const [input, setInput]             = useState("");
  const [typing, setTyping]           = useState(false);
  const [activeTopic, setActiveTopic] = useState("Сите"); // ← исто како searchTerm во FAQ
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
       behavior: "smooth",
      block: "nearest"
      });
  }, [messages, typing]);

  // Филтрирање — иста логика како filteredData во FAQ.jsx
  const filteredQuestions = QUICK_QUESTIONS.filter(
    (item) => activeTopic === "Сите" || item.cat === activeTopic
  );

  function sendQuestion(question) {
    setMessages((prev) => [...prev, { id: Date.now(), from: "user", text: question, time: now() }]);
    setInput("");
    setTyping(true);
    setTimeout(() => {
      const qa = QA_MAP[question];
      const botText = qa
        ? qa.answer
        : "За ова прашање, Ве молиме контактирајте ја надлежната институција или посетете го порталот uslugi.gov.mk.";
      setTyping(false);
      setMessages((prev) => [...prev, { id: Date.now() + 1, from: "bot", text: botText, time: now() }]);
    }, 900);
  }

  function handleSend() {
    const q = input.trim();
    if (q) sendQuestion(q);
  }

  return (
    <div style={styles.page}>
      <div style={styles.layout}>

        {/* LEFT SIDEBAR */}
        <aside style={styles.sidebar}>
          <div style={styles.sidebarSection}>
            <p style={styles.sidebarLabel}>БРЗИ ПРАШАЊА</p>

            {/* TOPIC BUTTONS — исто како tag buttons во FAQ.jsx */}
            <div style={styles.topicRow}>
              {TOPICS.map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTopic(t)}
                  style={{
                    padding: '5px 11px', borderRadius: '20px', border: '1px solid #eee',
                    fontFamily: 'inherit', fontSize: '12px', cursor: 'pointer',
                    fontWeight: '500', transition: '0.2s',
                    background: activeTopic === t ? brandBlue : 'white',
                    color:      activeTopic === t ? 'white'   : '#666',
                  }}
                >
                  {t}
                </button>
              ))}
            </div>

            {/* FILTERED QUESTIONS */}
            <div style={{ ...styles.qList, marginTop: 12 }}>
              {filteredQuestions.length > 0 ? filteredQuestions.map((item) => (
                <button key={item.q} style={styles.qBtn} onClick={() => sendQuestion(item.q)}>
                  <span style={styles.catBadge}>{item.cat}</span>
                  {item.q}
                </button>
              )) : (
                <p style={{ fontSize: 12, color: '#94a3b8', textAlign: 'center', padding: '10px 0', margin: 0 }}>
                  Нема прашања за оваа тема
                </p>
              )}
            </div>
          </div>
        </aside>

        {/* CHAT PANEL */}
        <div style={styles.chatPanel}>

          <div style={styles.chatHeader}>
            <div style={styles.botAvatar}><BotIcon /></div>
            <div>
              <div style={styles.botName}>е-Влада АИ Асистент</div>
              <div style={styles.botStatus}><span style={styles.dot} />Активен</div>
            </div>
            <button style={styles.resetBtn} onClick={() => setMessages([{
              id: Date.now(), from: "bot", time: now(),
              text: "Добредојдовте! Јас сум е-Влада АИ Асистент. Изберете прашање или напишете го вашето прашање на македонски.",
            }])}>↺ Ресетирај</button>
          </div>

          <div style={styles.msgArea}>
            {messages.map((msg) => (
              <div key={msg.id} style={{ display: "flex", justifyContent: msg.from === "user" ? "flex-end" : "flex-start", marginBottom: 14 }}>
                {msg.from === "bot"  && <div style={styles.botAvatarSmall}><BotIcon size={16} /></div>}
                <div style={msg.from === "user" ? styles.userBubble : styles.botBubble}>
                  <div>{formatAnswer(msg.text)}</div>
                  <div style={styles.timeStamp}>{msg.time}</div>
                </div>
                {msg.from === "user" && <div style={styles.userAvatarSmall}><UserIcon /></div>}
              </div>
            ))}
            {typing && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                <div style={styles.botAvatarSmall}><BotIcon size={16} /></div>
                <div style={styles.botBubble}><TypingDots /></div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div style={styles.inputArea}>
            <input
              style={styles.input}
              placeholder="Напишете прашање на македонски..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
            />
            <button style={styles.sendBtn} onClick={handleSend}><SendIcon /></button>
          </div>
          <div style={styles.disclaimer}>
            АИ одговорите се информативни и не претставуваат службена правна помош.
          </div>
        </div>
      </div>
    </div>
  );
}

function BotIcon({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect x="3" y="8" width="18" height="13" rx="3" fill="white" fillOpacity="0.9"/>
      <circle cx="9"  cy="14" r="2" fill={brandBlue}/>
      <circle cx="15" cy="14" r="2" fill={brandBlue}/>
      <rect x="10" y="4" width="4" height="4" rx="1" fill="white" fillOpacity="0.9"/>
      <line x1="12" y1="4" x2="12" y2="8" stroke="white" strokeWidth="1.5"/>
    </svg>
  );
}
function UserIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="8" r="4" fill="white" fillOpacity="0.9"/>
      <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="white" strokeWidth="2" strokeLinecap="round" fill="none"/>
    </svg>
  );
}
function SendIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M22 2L11 13" stroke="white" strokeWidth="2" strokeLinecap="round"/>
      <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" strokeWidth="2" strokeLinejoin="round" fill="none"/>
    </svg>
  );
}
function TypingDots() {
  return (
    <div style={{ display: "flex", gap: 5, padding: "4px 2px", alignItems: "center" }}>
      {[0, 1, 2].map((i) => (
        <span key={i} style={{
          width: 8, height: 8, borderRadius: "50%", background: "#94a3b8",
          display: "inline-block",
          animation: `bounce 1.2s ${i * 0.2}s infinite ease-in-out`,
        }} />
      ))}
      <style>{`@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}`}</style>
    </div>
  );
}

const styles = {
  page:            { minHeight: "100vh", background: lightBlue, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "'Segoe UI','Helvetica Neue',Arial,sans-serif", padding: "24px 16px", boxSizing: "border-box" },
  layout:          { display: "flex", gap: 20, width: "100%", maxWidth: 900, alignItems: "flex-start" },
  sidebar:         { width: 230, flexShrink: 0 },
  sidebarSection:  { background: "white", borderRadius: 14, padding: "20px 16px", boxShadow: "0 2px 12px rgba(0,0,0,0.07)" },
  sidebarLabel:    { fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.08em", margin: "0 0 12px 0" },
  topicRow:        { display: "flex", flexWrap: "wrap", gap: 6 },
  qList:           { display: "flex", flexDirection: "column", gap: 8 },
  qBtn:            { background: "none", border: "1px solid #e2e8f0", borderRadius: 8, padding: "9px 12px", textAlign: "left", fontSize: 13, color: "#334155", cursor: "pointer", lineHeight: 1.4, fontFamily: "inherit", display: "flex", flexDirection: "column", gap: 4 },
  catBadge:        { fontSize: 10, fontWeight: 700, color: brandBlue, background: lightBlue, padding: "2px 8px", borderRadius: 4, alignSelf: "flex-start" },
  chatPanel:       { flex: 1, background: "white", borderRadius: 16, boxShadow: "0 2px 16px rgba(0,0,0,0.09)", display: "flex", flexDirection: "column", overflow: "hidden", minHeight: 540 },
  chatHeader:      { display: "flex", alignItems: "center", gap: 12, padding: "16px 20px", borderBottom: "1px solid #f1f5f9" },
  botAvatar:       { width: 40, height: 40, background: `linear-gradient(135deg,${brandBlue},#4a6fa5)`, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 },
  botName:         { fontWeight: 700, fontSize: 15, color: "#1e293b" },
  botStatus:       { display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "#22c55e", fontWeight: 500 },
  dot:             { width: 7, height: 7, borderRadius: "50%", background: "#22c55e", display: "inline-block" },
  resetBtn:        { marginLeft: "auto", background: "none", border: "1px solid #e2e8f0", borderRadius: 8, padding: "6px 12px", fontSize: 13, color: "#64748b", cursor: "pointer", fontFamily: "inherit" },
  msgArea:         { flex: 1, overflowY: "auto", padding: "20px 20px 8px", display: "flex", flexDirection: "column", minHeight: 350, maxHeight: 420 },
  botAvatarSmall:  { width: 30, height: 30, background: `linear-gradient(135deg,${brandBlue},#4a6fa5)`, borderRadius: 9, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, alignSelf: "flex-end", marginRight: 8 },
  userAvatarSmall: { width: 30, height: 30, background: `linear-gradient(135deg,${brandBlue},#4a6fa5)`, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, alignSelf: "flex-end", marginLeft: 8 },
  botBubble:       { background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "14px 14px 14px 4px", padding: "12px 14px", maxWidth: "70%", fontSize: 13.5, color: "#334155", lineHeight: 1.6 },
  userBubble:      { background: `linear-gradient(135deg,${brandBlue},#4a6fa5)`, borderRadius: "14px 14px 4px 14px", padding: "12px 14px", maxWidth: "60%", fontSize: 13.5, color: "white", lineHeight: 1.6 },
  timeStamp:       { fontSize: 11, opacity: 0.55, marginTop: 4 },
  inputArea:       { display: "flex", alignItems: "center", gap: 10, padding: "14px 16px 8px", borderTop: "1px solid #f1f5f9" },
  input:           { flex: 1, border: "1px solid #e2e8f0", borderRadius: 10, padding: "10px 14px", fontSize: 14, color: "#334155", outline: "none", fontFamily: "inherit", background: "#f8fafc" },
  sendBtn:         { background: `linear-gradient(135deg,${brandBlue},#4a6fa5)`, border: "none", borderRadius: 10, width: 42, height: 42, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", flexShrink: 0 },
  disclaimer:      { fontSize: 11, color: "#94a3b8", textAlign: "center", padding: "4px 16px 14px" },
};
