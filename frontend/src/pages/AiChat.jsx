import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Footer from '../components/Footer';

const fallback = {
    'пасош': 'За добивање пасош потребно е да поднесете барање во МВР. Потребни документи: лична карта, уплатница (1.100 ден.), 1 фотографија. Рок: 30 дена, итна постапка: 5 работни дена.',
    'данок': 'Рокот за годишната даночна пријава е 15 март. За помош: ujp.gov.mk или 0800 33 44.',
    'лична': 'За лична карта: извод од МКР + уплатница (300 ден.) + претходна лична карта. Рок: 15 работни дена.',
    'возачка': 'За возачка дозвола: лична карта + лекарско уверение + положен испит. Такса: 1.200 ден.',
    'социјал': 'Поднесете барање во Центарот за социјална работа. Условите зависат од приходите на домаќинството.',
    'термин': 'За закажување термин во МВР: mvr.gov.mk или +389 02 3117 222. Работно време: 08:00–16:00.',
    'документи': 'Изберете: пасош, лична карта, возачка дозвола или друг документ?',
    'даноци': 'УЈП нуди онлајн услуги на ujp.gov.mk – пријави, проверка и плаќање онлајн.',
    'здравство': 'За здравствени услуги: матичен лекар или ФЗОМ на 02 3200 400.',
    'образование': 'За образовни услуги: mon.gov.mk или studentski.finki.ukim.mk.',
    'плаќање': 'Таксите може да се платат онлајн преку e-плаќање на na.mk или преку банка.',
    'локација': 'За локации на институции посетете vlada.gov.mk.',
};

const quickQuestions = [
    'Како да добијам пасош?',
    'Рок за даночна пријава',
    'Документи за лична карта',
    'Возачка дозвола – барање',
    'Социјална помош услови',
    'Закажи термин во МВР',
];

const topics = ['Документи', 'Даноци', 'Социјала', 'Локации', 'Плаќање', 'Здравство', 'Образование'];

function getTime() {
    return new Date().toLocaleTimeString('mk', { hour: '2-digit', minute: '2-digit' });
}

const initialMessages = [
    { type: 'bot', text: 'Добар ден! Јас сум АИ Асистентот на порталот БрзиУслуги. Подготвен сум да одговорам на вашите прашања за јавните услуги – документи, даноци, социјала, и многу повеќе.', time: getTime() },
    { type: 'bot', text: 'Како можам да ви помогнам денес?', time: getTime() },
];

function AiChat() {
    const navigate = useNavigate();
    const [messages, setMessages] = useState(initialMessages);
    const [input, setInput] = useState('');
    const [typing, setTyping] = useState(false);
    const [services, setServices] = useState([]);
    const [selectedServiceId, setSelectedServiceId] = useState('');
    const [selectedFields, setSelectedFields] = useState({
        full_name: true,
        email: true,
        embg: true,
        address: false,
        phone_number: false,
        gender: false,
    });
    const [docStatus, setDocStatus] = useState('');
    const [autoFillLoading, setAutoFillLoading] = useState(false);
    const [selectedFormat, setSelectedFormat] = useState('txt');
    const msgsRef = useRef(null);

    useEffect(() => {
        if (msgsRef.current) {
            msgsRef.current.scrollTop = msgsRef.current.scrollHeight;
        }
    }, [messages, typing]);

    useEffect(() => {
        const loadServices = async () => {
            try {
                const response = await fetch('http://127.0.0.1:8000/services');
                const data = await response.json();
                if (response.ok && Array.isArray(data)) {
                    setServices(data);
                    if (data.length > 0) {
                        setSelectedServiceId(data[0].id);
                    }
                }
            } catch {
                setDocStatus('Не успеа да се вчита листата со услуги.');
            }
        };

        loadServices();
    }, []);


    const addMsg = (msg) => setMessages(prev => [...prev, msg]);

    const downloadBlob = (blob, filename) => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
    };

    const handleAutoFillDownload = async () => {
        const token = localStorage.getItem('token');
        if (!token) {
            setDocStatus('Потребна е најава за авто-полнење.');
            return;
        }
        if (!selectedServiceId) return;

        const selectedFieldList = Object.entries(selectedFields)
            .filter(([, enabled]) => enabled)
            .map(([key]) => key);

        setAutoFillLoading(true);
        setDocStatus('');
        try {
            const response = await fetch(`http://127.0.0.1:8000/service-document-templates/${selectedServiceId}/auto-fill?format=${encodeURIComponent(selectedFormat)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({ selected_fields: selectedFieldList }),
            });

            if (!response.ok) {
                if (response.status === 404) {
                    setDocStatus('Оваа услуга нема достапен документ за пополнување.');
                } else {
                    setDocStatus('Неуспешно авто-полнење на документот.');
                }
                return;
            }

            const blob = await response.blob();
            const ext = selectedFormat === 'pdf' ? 'pdf' : selectedFormat === 'docx' ? 'docx' : 'txt';
            const selectedServiceName = services.find(s => s.id === selectedServiceId)?.name || 'document';
            downloadBlob(blob, `${selectedServiceName}_пополнета_пријава.${ext}`);
            setDocStatus('✓ Документот е преземен.');
        } catch {
            setDocStatus('Грешка при авто-полнење.');
        } finally {
            setAutoFillLoading(false);
        }
    };

    const send = async (text) => {
        const query = (text || input).trim();
        if (!query) return;
        setInput('');
        addMsg({ type: 'user', text: query, time: getTime() });
        setTyping(true);

        try {
            const r = await fetch('http://127.0.0.1:8000/chat/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: query }),
            });
            const data = await r.json();
            setTyping(false);
            if (r.ok) {
                addMsg({ type: 'bot', text: data.raw_response, time: getTime() });
            } else {
                addMsg({ type: 'bot', text: 'Нема резултат. Обидете се со поинакво прашање.', time: getTime() });
            }
        } catch {
            setTyping(false);
            const lower = query.toLowerCase();
            let reply = 'За подетални информации посетете gov.mk или јавете се на 0800 100 200 (бесплатно).';
            for (const [k, v] of Object.entries(fallback)) {
                if (lower.includes(k)) { reply = v; break; }
            }
            addMsg({ type: 'bot', text: reply, time: getTime() });
        }
    };

    const resetChat = () => setMessages([
        { type: 'bot', text: 'Добар ден! Јас сум АИ Асистентот на порталот БрзиУслуги. Подготвен сум да одговорам на вашите прашања за јавните услуги – документи, даноци, социјала, и многу повеќе.', time: getTime() },
        { type: 'bot', text: 'Како можам да ви помогнам денес?', time: getTime() },
    ]);

    return (
        <div style={{ minHeight: '100vh', background: '#F4F6FA', display: 'flex', flexDirection: 'column' }}>

            {/* Hero */}
            <div style={{ background: 'linear-gradient(150deg,#0f2044 0%,#1B3A6B 55%,#1a4a8a 100%)', padding: '52px 40px 72px', textAlign: 'center' }}>
                <div style={{ width: 68, height: 68, background: 'rgba(255,255,255,.1)', borderRadius: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px', border: '1px solid rgba(255,255,255,.15)', position: 'relative' }}>
                    <div style={{ width: 46, height: 46, background: '#D4A017', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>🤖</div>
                    <div style={{ position: 'absolute', top: -4, right: -4, width: 14, height: 14, background: '#22c55e', borderRadius: '50%', border: '2.5px solid #1B3A6B' }} />
                </div>
                <h2 style={{ color: '#fff', fontSize: '2.2rem', fontWeight: 800, margin: '0 0 10px' }}>АИ Асистент</h2>
                <p style={{ color: '#D4A017', margin: 0, fontSize: '0.9rem' }}>Вештачка интелигенција за јавни услуги на БрзиУслуги</p>
            </div>

            {/* Main */}
            <div style={{ maxWidth: 1080, margin: '-44px auto 56px', padding: '0 28px', display: 'grid', gridTemplateColumns: '320px 1fr', gap: 22, position: 'relative', zIndex: 10 }}>

                {/* Sidebar */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                    <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #e2e8f0', overflow: 'hidden' }}>
                        <div style={{ padding: '14px 18px 12px', fontSize: '0.7rem', fontWeight: 700, color: '#64748b', letterSpacing: '0.12em', textTransform: 'uppercase', borderBottom: '1px solid #e2e8f0' }}>Брзи Прашања</div>
                        <div style={{ padding: 10 }}>
                            {quickQuestions.map(q => (
                                <button key={q} onClick={() => send(q)} style={{ width: '100%', background: 'none', border: '1.5px solid #e2e8f0', borderRadius: 10, padding: '11px 14px', textAlign: 'left', fontSize: '0.82rem', color: '#2a3a5e', cursor: 'pointer', fontWeight: 500, marginBottom: 6, display: 'block' }}>
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                    
                    <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #e2e8f0', overflow: 'hidden' }}>
                        <div style={{ padding: '14px 18px 12px', fontSize: '0.7rem', fontWeight: 700, color: '#64748b', letterSpacing: '0.12em', textTransform: 'uppercase', borderBottom: '1px solid #e2e8f0' }}>Документи</div>
                        <div style={{ padding: 14, display: 'grid', gap: 10 }}>
                            <p style={{ margin: 0, fontSize: '0.78rem', color: '#64748b', lineHeight: 1.5 }}>
                                Изберете услуга и полиња, па кликнете „Автоматски пополни" за да добиете пополнет документ со Вашите податоци.
                            </p>
                            <select
                                value={selectedServiceId}
                                onChange={(e) => setSelectedServiceId(parseInt(e.target.value))}
                                style={{ width: '100%', border: '1.5px solid #e2e8f0', borderRadius: 10, padding: '10px 12px', fontSize: '0.82rem', background: '#f8fafc' }}
                            >
                                {services.map((service) => (
                                    <option key={service.id} value={service.id}>
                                        {service.name}
                                    </option>
                                ))}
                            </select>

                            <div style={{ display: 'grid', gap: 6, fontSize: '0.8rem', color: '#334155' }}>
                                <label><input type="checkbox" checked={selectedFields.full_name} onChange={(e) => setSelectedFields(prev => ({ ...prev, full_name: e.target.checked }))} /> Име и презиме</label>
                                <label><input type="checkbox" checked={selectedFields.email} onChange={(e) => setSelectedFields(prev => ({ ...prev, email: e.target.checked }))} /> Е-маил</label>
                                <label><input type="checkbox" checked={selectedFields.embg} onChange={(e) => setSelectedFields(prev => ({ ...prev, embg: e.target.checked }))} /> ЕМБГ</label>
                                <label><input type="checkbox" checked={selectedFields.address} onChange={(e) => setSelectedFields(prev => ({ ...prev, address: e.target.checked }))} /> Адреса</label>
                                <label><input type="checkbox" checked={selectedFields.phone_number} onChange={(e) => setSelectedFields(prev => ({ ...prev, phone_number: e.target.checked }))} /> Телефон</label>
                                <label><input type="checkbox" checked={selectedFields.gender} onChange={(e) => setSelectedFields(prev => ({ ...prev, gender: e.target.checked }))} /> Пол</label>
                            </div>

                            <select
                                value={selectedFormat}
                                onChange={(e) => setSelectedFormat(e.target.value)}
                                style={{ width: '100%', border: '1.5px solid #e2e8f0', borderRadius: 10, padding: '8px 10px', fontSize: '0.82rem', background: '#fff' }}
                            >
                                <option value="txt">TXT</option>
                                <option value="pdf">PDF</option>
                                <option value="docx">Word (.docx)</option>
                            </select>

                            <button
                                onClick={handleAutoFillDownload}
                                disabled={autoFillLoading}
                                style={{ width: '100%', background: autoFillLoading ? '#94a3b8' : 'linear-gradient(135deg,#1B3A6B 0%,#2563eb 100%)', color: '#fff', border: 'none', padding: '10px 12px', borderRadius: 10, fontSize: '0.82rem', fontWeight: 700, cursor: autoFillLoading ? 'not-allowed' : 'pointer', transition: 'background 0.2s' }}
                            >
                                {autoFillLoading ? '⏳ Се генерира...' : '✦ Автоматски пополни'}
                            </button>

                            {docStatus && <div style={{ fontSize: '0.75rem', color: '#64748b', lineHeight: 1.4 }}>{docStatus}</div>}
                        </div>
                    </div>
                </div>

                {/* Chat */}
                <div style={{ background: '#fff', borderRadius: 16, border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', height: 600 }}>
                    {/* Chat header */}
                    <div style={{ padding: '14px 18px', borderBottom: '1px solid #e2e8f0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
                            <div style={{ width: 40, height: 40, background: '#1B3A6B', borderRadius: 11, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 17, position: 'relative' }}>
                                🤖
                                <div style={{ position: 'absolute', bottom: -2, right: -2, width: 11, height: 11, background: '#22c55e', borderRadius: '50%', border: '2px solid #fff' }} />
                            </div>
                            <div>
                                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f2044' }}>БрзиУслуги АИ Асистент</div>
                                <div style={{ fontSize: '0.75rem', color: '#22c55e', fontWeight: 600 }}>● Активен</div>
                            </div>
                        </div>
                        <button onClick={resetChat} style={{ background: 'none', border: '1.5px solid #e2e8f0', borderRadius: 8, padding: '6px 13px', fontSize: '0.75rem', color: '#64748b', cursor: 'pointer' }}>↺ Ресетирај</button>
                    </div>

                    {/* Messages */}
                    <div ref={msgsRef} style={{ flex: 1, overflowY: 'auto', padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
                        {messages.map((msg, i) => (
                            <div key={i} style={{ display: 'flex', gap: 9, alignItems: 'flex-end', flexDirection: msg.type === 'user' ? 'row-reverse' : 'row' }}>
                                <div style={{ width: 30, height: 30, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, flexShrink: 0, background: msg.type === 'user' ? '#D4A017' : '#1B3A6B' }}>
                                    {msg.type === 'user' ? '👤' : '🤖'}
                                </div>
                                <div>
                                    {msg.type === 'faq' ? (
                                        <div style={{ background: '#f0f4ff', borderRadius: 13, padding: '12px 15px', fontSize: '0.82rem', lineHeight: 1.65, maxWidth: 400 }}>
                                            <div style={{ fontWeight: 700, color: '#1B3A6B', marginBottom: 5 }}>❓ {msg.question}</div>
                                            <div>{msg.answer}</div>
                                            <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: 6, fontStyle: 'italic' }}>Доверба: {msg.confidence} ({msg.score?.toFixed(3)})</div>
                                        </div>
                                    ) : (
                                        <div style={{ maxWidth: 340, padding: '11px 15px', borderRadius: 13, fontSize: '0.83rem', lineHeight: 1.65, background: msg.type === 'user' ? '#1B3A6B' : '#f0f4ff', color: msg.type === 'user' ? '#fff' : '#1e293b' }}>
                                            {msg.text}
                                        </div>
                                    )}
                                    <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: 3, textAlign: msg.type === 'user' ? 'right' : 'left' }}>{msg.time}</div>
                                </div>
                            </div>
                        ))}
                        {typing && (
                            <div style={{ display: 'flex', gap: 9, alignItems: 'flex-end' }}>
                                <div style={{ width: 30, height: 30, borderRadius: 9, background: '#1B3A6B', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13 }}>🤖</div>
                                <div style={{ background: '#f0f4ff', borderRadius: 13, padding: '10px 14px' }}>
                                    <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                                        {[0, 1, 2].map(i => (
                                            <div key={i} style={{ width: 7, height: 7, background: '#94a3b8', borderRadius: '50%', animation: `bounce 1.2s ${i * 0.2}s infinite` }} />
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Input */}
                    <div style={{ padding: '14px 16px', borderTop: '1px solid #e2e8f0' }}>
                        <div style={{ display: 'flex', gap: 9, alignItems: 'center' }}>
                            <input
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={e => e.key === 'Enter' && send()}
                                placeholder="Напишете прашање на македонски..."
                                style={{ flex: 1, border: '1.5px solid #e2e8f0', borderRadius: 11, padding: '11px 15px', fontSize: '0.85rem', outline: 'none', background: '#f8fafc' }}
                            />
                            <button onClick={() => send()} style={{ width: 44, height: 44, background: '#1B3A6B', border: 'none', borderRadius: 11, cursor: 'pointer', color: '#fff', fontSize: 16, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>➤</button>
                        </div>
                        <p style={{ fontSize: '0.7rem', color: '#64748b', textAlign: 'center', marginTop: 7 }}>АИ одговорите се информативни и не претставуваат службена правна помош.</p>
                    </div>
                </div>
            </div>
            <Footer />
        </div>
    );
}

export default AiChat;