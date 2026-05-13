import React, {useState, useEffect} from 'react';
import {useNavigate} from 'react-router-dom';
import '../index.css';
import Footer from '../components/Footer';

const TICKER_ITEMS = [
    { label: 'Пасош', icon: '🪪', color: '#1B3A6B' },
    { label: 'Лична карта', icon: '🪪', color: '#1B3A6B' },
    { label: 'Возачка дозвола', icon: '🚗', color: '#1B3A6B' },
    { label: 'Даночна пријава', icon: '📋', color: '#CE2028' },
    { label: 'Социјална помош', icon: '🤝', color: '#D4A017' },
    { label: 'Здравствена заштита', icon: '🏥', color: '#166534' },
    { label: 'Пензиски придонес', icon: '📑', color: '#7c3aed' },
    { label: 'Регистрација на фирма', icon: '🏢', color: '#0369a1' },
    { label: 'Извод од МКР', icon: '📄', color: '#1B3A6B' },
    { label: 'Плаќање такси', icon: '💳', color: '#CE2028' },
    { label: 'Закажи термин', icon: '📅', color: '#166534' },
    { label: 'Образовни услуги', icon: '🎓', color: '#7c3aed' },
];

function Home() {
    const [showModal, setShowModal] = useState(false);
    const [authMode, setAuthMode] = useState("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [fullName, setFullName] = useState("");
    const [address, setAddress] = useState("");
    const [phoneNumber, setPhoneNumber] = useState("");
    const [gender, setGender] = useState("");
    const [authError, setAuthError] = useState("");
    const navigate = useNavigate();

    const handleAuth = async (e) => {
        e.preventDefault();
        setAuthError("");
        const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
        const payload = authMode === "login"
            ? {username: email, password: password}
            : {email, full_name: fullName, address, phone_number: phoneNumber, gender, password};
        try {
            const response = await fetch(`http://127.0.0.1:8000${endpoint}`, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (response.ok) {
                if (authMode === "login") {
                    localStorage.setItem("token", data.access_token);
                    setShowModal(false);
                    window.location.reload();
                } else {
                    setAuthMode("login");
                    setAuthError("");
                }
            } else {
                setAuthError(data.detail || "Проверете ги внесените податоци.");
            }
        } catch {
            setAuthError("Грешка при поврзување со серверот.");
        }
    };

    return (
        <>
            {/* --- AUTH MODAL --- */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="close-btn" onClick={() => setShowModal(false)}>&times;</button>
                        <div className="modal-tabs">
                            <button className={authMode === "login" ? "active" : ""} onClick={() => { setAuthMode("login"); setAuthError(""); }}>Најава</button>
                            <button className={authMode === "register" ? "active" : ""} onClick={() => { setAuthMode("register"); setAuthError(""); }}>Регистрација</button>
                        </div>
                        <form onSubmit={handleAuth} className="auth-form">
                            {authMode === "register" && (
                                <>
                                    <input type="text" placeholder="Целосно име" value={fullName} onChange={(e) => setFullName(e.target.value)} required />
                                    <input type="text" placeholder="Адреса (опционално)" value={address} onChange={(e) => setAddress(e.target.value)} />
                                    <input type="text" placeholder="Телефон (опционално)" value={phoneNumber} onChange={(e) => setPhoneNumber(e.target.value)} />
                                    <select value={gender} onChange={(e) => setGender(e.target.value)} style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd", fontSize: "0.95rem"}}>
                                        <option value="">Пол (опционално)</option>
                                        <option value="Машки">Машки</option>
                                        <option value="Женски">Женски</option>
                                    </select>
                                </>
                            )}
                            <input type="email" placeholder="Е-пошта" value={email} onChange={(e) => setEmail(e.target.value)} required />
                            <input type="password" placeholder="Лозинка" value={password} onChange={(e) => setPassword(e.target.value)} required />
                            {authError && (
                                <div style={{background: "#FFF1F2", border: "1px solid #fecdd3", borderRadius: 6, padding: "10px 12px", color: "#CE2028", fontSize: "0.84rem"}}>
                                    {authError}
                                </div>
                            )}
                            <button type="submit" className="btn-link5" style={{width: "100%", marginTop: "8px"}}>
                                {authMode === "login" ? "Влези" : "Регистрирај се"}
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* --- HERO SECTION --- */}
            <div className="fourth-container">
                <div className="box-inside">
                    <h1 style={{color: "white"}}>Добредојдовте на <span style={{color: "rgb(212, 160, 23)"}}>БрзиУслуги</span></h1>
                    <p style={{color: "rgb(147, 197, 253)"}}>Вашиот дигитален портал за сите владини услуги. Едноставно и достапно 24/7.</p>
                    <div className="buttons-inside">
                        <button className="btn-link6" onClick={() => navigate('/services')}>Разгледај Услуги &rarr;</button>
                        <button className="btn-link7" onClick={() => navigate("/chat")}>АИ Асистент</button>
                    </div>
                </div>
            </div>

            {/* --- ANIMATED SERVICE TICKER --- */}
            <div style={{background: "#0f2044", padding: "14px 0", overflow: "hidden", borderTop: "1px solid rgba(255,255,255,0.08)", borderBottom: "1px solid rgba(255,255,255,0.08)"}}>
                <div className="ticker-track">
                    {[...TICKER_ITEMS, ...TICKER_ITEMS].map((item, i) => (
                        <div
                            key={i}
                            onClick={() => navigate('/services')}
                            style={{
                                display: "inline-flex", alignItems: "center", gap: 8,
                                marginRight: 40, cursor: "pointer",
                                color: "#ececf0", fontSize: "0.85rem", fontWeight: 500,
                                whiteSpace: "nowrap",
                            }}
                        >
                            <span style={{
                                background: `${item.color}22`,
                                border: `1px solid ${item.color}55`,
                                borderRadius: 6, padding: "3px 10px",
                                color: "#D4A017", fontSize: "0.78rem"
                            }}>
                                {item.icon} {item.label}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* --- STATS --- */}
            <div className="fifth-container">
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>2.1М+</p><p style={{color: "rgb(147, 197, 253)"}}>Граѓани</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>150+</p><p style={{color: "rgb(147, 197, 253)"}}>Услуги</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>24/7</p><p style={{color: "rgb(147, 197, 253)"}}>Пристап</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>98%</p><p style={{color: "rgb(147, 197, 253)"}}>Задоволство</p></div>
            </div>

            {/* --- SERVICES CAROUSEL --- */}
            <div className="seventh-container">
                <div className="la-texta">
                    <h2 style={{color: "rgb(27, 58, 107)"}}>Нашите Услуги</h2>
                </div>
                <div style={{overflow: "hidden", padding: "8px 0 16px"}}>
                    <div style={{display: "flex", gap: 20, overflowX: "auto", paddingBottom: 8, scrollbarWidth: "none"}}>
                        {[
                            {title: "Лични Документи", desc: "Пасош, лична карта, возачка дозвола", color: "#1B3A6B", icon: "🪪"},
                            {title: "Даноци и финансии", desc: "Даночна пријава, УЈП услуги, плаќања", color: "#CE2028", icon: "📋"},
                            {title: "Социјални Услуги", desc: "Бенефиции, пензии, здравствена заштита", color: "#D4A017", icon: "🤝"},
                            {title: "Локации", desc: "Канцеларии, општини, институции", color: "#166534", icon: "📍"},
                            {title: "АИ Асистент", desc: "Брзи одговори на вашите прашања", color: "#7c3aed", icon: "🤖"},
                            {title: "Образование", desc: "Студентски услуги, матуранти", color: "#0369a1", icon: "🎓"},
                        ].map((card, i) => (
                            <div key={i} className="carousel-card" onClick={() => navigate('/services')}>
                                <div style={{fontSize: 32, marginBottom: 12}}>{card.icon}</div>
                                <h3 style={{color: card.color, fontSize: "1rem", marginBottom: 6}}>{card.title}</h3>
                                <p style={{color: "#64748b", fontSize: "0.82rem", lineHeight: 1.5, marginBottom: 16}}>{card.desc}</p>
                                <span style={{color: card.color, fontSize: "0.78rem", fontWeight: 700}}>Дознај повеќе →</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* --- INFO SECTION --- */}
            <div className="eigth-container">
                <div className="eigth-text">
                    <p className="zosto">Зошто БрзиУслуги?</p>
                    <h2 style={{color: "rgb(27, 58, 107)"}}>Дигитална трансформација на јавните услуги</h2>
                    <p>БрзиУслуги е официјалниот портал на Владата на Република Северна Македонија кој овозможува брз, лесен и безбеден пристап до над 150 јавни услуги директно од дома.</p>
                    <p>✓ Поднесување барања без чекање на ред</p>
                    <p>✓ Проверка на статус на барање во реално време</p>
                    <p>✓ Безбедно плаќање на такси и придонеси</p>
                    <p>✓ АИ асистент за брза помош</p>
                    <button onClick={() => navigate("/services")}>Почни сега &rarr;</button>
                </div>
                <div className="eigth-image">
                    <img
                        style={{height: "400px", width: "700px", borderRadius: "20px", objectFit: "cover"}}
                        src="https://images.unsplash.com/photo-1746044060948-ce76677390f2?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&w=800"
                        alt="Digital Government Services"
                    />
                </div>
            </div>

            <Footer />
        </>
    );
}

export default Home;