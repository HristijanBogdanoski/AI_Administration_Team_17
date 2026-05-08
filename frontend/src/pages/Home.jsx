import React, {useState, useEffect} from 'react';
import {useNavigate} from 'react-router-dom';
import '../index.css';
import Footer from '../components/Footer';

function Home() {
    const [showModal, setShowModal] = useState(false);
    const [showAuthChoice, setShowAuthChoice] = useState(false);
    const [authMode, setAuthMode] = useState("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [fullName, setFullName] = useState("");
    const [address, setAddress] = useState("");
    const [phoneNumber, setPhoneNumber] = useState("");
    const [gender, setGender] = useState("");
    const [choiceEmail, setChoiceEmail] = useState("");
    const [showChat, setShowChat] = useState(false);
    const [chatMessages, setChatMessages] = useState([]);
    const [chatInput, setChatInput] = useState("");
    const navigate = useNavigate();

    /* global google */
    useEffect(() => {
        if (window.google) {
            google.accounts.id.initialize({
                // Note: You can replace this with a real Client ID from Google Cloud Console
                client_id: "YOUR_GOOGLE_CLIENT_ID.apps.googleusercontent.com",
                callback: (response) => {
                    console.log("Google JWT Response:", response.credential);
                    // As requested, no login functionality is added here
                    alert("Успешно избравте Google профил (Прототип режим).");
                }
            });
        }
    }, []);

    const handleGoogleLogin = () => {
        if (window.google) {
            // This triggers the official Google floating "One Tap" or account picker
            google.accounts.id.prompt();
        } else {
            console.error("Google script not loaded");
        }
    };

    const handleAuth = async (e) => {
        e.preventDefault();
        const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
        const payload = authMode === "login"
            ? {username: email, password: password} // We map your 'email' state to the 'username' key
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
                } else {
                    alert("Успешна регистрација! Сега можете да се најавите.");
                    setAuthMode("login");
                }
            } else {
                alert("Грешка: " + (data.detail || "Проверете ги податоците"));
            }
        } catch (err) {
            console.error("Грешка при поврзување со бекендот");
        }
    };

    const handleLogout = () => {
        localStorage.removeItem("token");
        window.location.reload();
    };

    const handleChoiceEmailContinue = () => {
        if (!choiceEmail.trim()) return;
        setShowAuthChoice(false);
        setEmail(choiceEmail);
        setAuthMode("register");
        setShowModal(true);
    };

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!chatInput.trim()) return;
        const userMessage = {type: "user", text: chatInput};
        setChatMessages(prev => [...prev, userMessage]);
        const currentInput = chatInput;
        setChatInput("");
        try {
            const response = await fetch("http://127.0.0.1:8000/faq/search", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query: currentInput, top_k: 3})
            });
            const data = await response.json();
            if (response.ok) {
                const botMessages = data.results.map(result => ({
                    type: "bot",
                    question: result.question,
                    answer: result.answer,
                    confidence: result.confidence,
                    score: result.score
                }));
                setChatMessages(prev => [...prev, ...botMessages]);
            } else {
                setChatMessages(prev => [...prev, {type: "bot", text: "Грешка при добивање на одговор."}]);
            }
        } catch (err) {
            console.error("Грешка при поврзување со серверот.");
            setChatMessages(prev => [...prev, {type: "bot", text: "Грешка при поврзување со серверот."}]);
        }
    };

    return (
        <>
            {/* --- AUTH CHOICE MODAL --- */}
            {showAuthChoice && (
                <div className="modal-overlay" onClick={() => setShowAuthChoice(false)}>
                    <div
                        onClick={(e) => e.stopPropagation()}
                        style={{
                            background: "#000",
                            color: "#fff",
                            borderRadius: "16px",
                            padding: "36px 32px",
                            width: "100%",
                            maxWidth: "400px",
                            position: "relative",
                            textAlign: "center"
                        }}
                    >
                        <button
                            onClick={() => setShowAuthChoice(false)}
                            style={{
                                position: "absolute", top: "16px", right: "16px",
                                background: "none", border: "none", color: "#fff",
                                fontSize: "1.4rem", cursor: "pointer", lineHeight: 1
                            }}
                        >&times;</button>
                        11

                        <p style={{
                            fontSize: "0.8rem",
                            color: "#aaa",
                            marginBottom: "6px",
                            textTransform: "uppercase",
                            letterSpacing: "0.05em"
                        }}>
                            Create an account to collaborate on
                        </p>
                        <h3 style={{color: "#fff", fontSize: "1.2rem", marginBottom: "28px", fontWeight: "600"}}>
                            "Government Chatbot Home FAQ"
                        </h3>

                        {/* Updated Continue with Google */}
                        <button
                            onClick={handleGoogleLogin}
                            style={{
                                width: "100%", padding: "12px 16px", marginBottom: "20px",
                                background: "#1a1a1a", border: "1px solid #333", borderRadius: "8px",
                                color: "#fff", fontSize: "0.95rem", cursor: "pointer",
                                display: "flex", alignItems: "center", justifyContent: "center", gap: "10px"
                            }}
                        >
                            <img src="https://www.google.com/favicon.ico" alt="Google"
                                 style={{width: "18px", height: "18px"}}/>
                            Continue with Google
                        </button>

                        <div style={{display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px"}}>
                            <hr style={{flex: 1, borderColor: "#333", borderTop: "1px solid #333"}}/>
                            <span style={{color: "#666", fontSize: "0.85rem"}}>or</span>
                            <hr style={{flex: 1, borderColor: "#333", borderTop: "1px solid #333"}}/>
                        </div>

                        <input
                            type="email"
                            placeholder="Email address"
                            value={choiceEmail}
                            onChange={(e) => setChoiceEmail(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleChoiceEmailContinue()}
                            style={{
                                width: "100%", padding: "11px 14px", marginBottom: "12px",
                                background: "#1a1a1a", border: "1px solid #333", borderRadius: "8px",
                                color: "#fff", fontSize: "0.95rem", boxSizing: "border-box",
                                outline: "none"
                            }}
                        />

                        <button
                            onClick={handleChoiceEmailContinue}
                            style={{
                                width: "100%", padding: "12px 16px",
                                background: "#fff", border: "none", borderRadius: "8px",
                                color: "#000", fontSize: "0.95rem", fontWeight: "600",
                                cursor: "pointer"
                            }}
                        >
                            Continue with Email
                        </button>
                    </div>
                </div>
            )}

            {/* --- AUTH MODAL --- */}
            {showModal && (
                <div className="modal-overlay" onClick={() => setShowModal(false)}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="close-btn" onClick={() => setShowModal(false)}>&times;</button>
                        <div className="modal-tabs">
                            <button className={authMode === "login" ? "active" : ""}
                                    onClick={() => setAuthMode("login")}>Најава
                            </button>
                            <button className={authMode === "register" ? "active" : ""}
                                    onClick={() => setAuthMode("register")}>Регистрација
                            </button>
                        </div>
                        <form onSubmit={handleAuth} className="auth-form">
                            {authMode === "register" && (
                                <>
                                <input
                                    type="text"
                                    placeholder="Целосно име"
                                    value={fullName}
                                    onChange={(e) => setFullName(e.target.value)}
                                    required
                                />
                                <input
                                    type="text"
                                    placeholder="Адреса (опционално)"
                                    value={address}
                                    onChange={(e) => setAddress(e.target.value)}
                                />
                                <input
                                    type="text"
                                    placeholder="Телефон (опционално)"
                                    value={phoneNumber}
                                    onChange={(e) => setPhoneNumber(e.target.value)}
                                />
                                <input
                                    type="text"
                                    placeholder="Пол (опционално)"
                                    value={gender}
                                    onChange={(e) => setGender(e.target.value)}
                                />
                                </>
                            )}
                            <input
                                type="email"
                                placeholder="Е-пошта"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                            <input
                                type="password"
                                placeholder="Лозинка"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                            <button type="submit" className="btn-link5" style={{width: "100%", marginTop: "15px"}}>
                                {authMode === "login" ? "Влези" : "Регистрирај се"}
                            </button>
                        </form>
                    </div>
                </div>
            )}


            {/* --- HERO SECTION --- */}
            <div className="fourth-container">
                <div className="box-inside">
                    <h1 style={{color: "white"}}>Добредојдовте на <span
                        style={{color: "rgb(212, 160, 23)"}}>е-Влада</span></h1>
                    <p style={{color: "rgb(147, 197, 253)"}}>Вашиот дигитален портал за сите валидни услуги. Брзо, лесно
                        и достапно 24/7.</p>
                    <div className="buttons-inside">
                        <button className="btn-link6" onClick={() => navigate('/services')}>Разгледај
                            Услуги &rarr;</button>
                        <button
                            className={`btn-link7 ${location.pathname === "/chat" ? "active" : ""}`}
                            onClick={() => navigate("/chat")}
                        >
                            АИ Асистент
                        </button>
                    </div>
                </div>
            </div>

            {/* --- STATS --- */}
            <div className="fifth-container">
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>2.1М+</p><p
                    style={{color: "rgb(147, 197, 253)"}}>Граѓани</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>150+</p><p
                    style={{color: "rgb(147, 197, 253)"}}>Услуги</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>24/7</p><p
                    style={{color: "rgb(147, 197, 253)"}}>Пристап</p></div>
                <div><p style={{color: "rgb(212, 160, 23)", fontWeight: "700", fontSize: "1.5rem"}}>98%</p><p
                    style={{color: "rgb(147, 197, 253)"}}>Задоволство</p></div>
            </div>

            {/* --- SERVICES GRID --- */}
            <div className="seventh-container">
                <div className="la-texta">
                    <h2 style={{color: "rgb(27, 58, 107)"}}>Нашите Услуги</h2>
                </div>

                <div className="cards-wrapper">
                    <div className="service-card">
                        <h3>Лични Документи</h3>
                        <p>Пасош, лична карта, возачка дозвола</p>
                        <button
                            className="card-btn"
                            style={{cursor: "pointer"}}
                            onClick={() => navigate("/services")}
                        >
                            Дознај повеќе
                        </button>
                    </div>

                    <div className="service-card">
                        <h3>Даноци и финансии</h3>
                        <p>Даночна пријава, УЈП услуги, плаќања</p>
                        <button
                            className="card-btn"
                            style={{color: "red", cursor: "pointer"}}
                            onClick={() => navigate("/services")}
                        >
                            Дознај повеќе
                        </button>
                    </div>

                    <div className="service-card">
                        <h3>Социјални Услуги</h3>
                        <p>Бенефиции, пензии, здравствена заштита</p>
                        <button
                            className="card-btn"
                            style={{color: "rgb(212, 160, 23)", cursor: "pointer"}}
                            onClick={() => navigate("/services")}
                        >
                            Дознај повеќе
                        </button>
                    </div>

                    <div className="service-card">
                        <h3>Локации</h3>
                        <p>Канцеларии, општини, институции</p>
                        <button
                            className="card-btn"
                            style={{color: "rgb(22, 101, 52)", cursor: "pointer"}}
                            onClick={() => navigate("/services")}
                        >
                            Дознај повеќе
                        </button>
                    </div>
                </div>
            </div>

            {/* --- INFO SECTION --- */}
            <div className="eigth-container">
                <div className="eigth-text">
                    <p className="zosto">Зошто е-Влада?</p>
                    <h2 style={{color: "rgb(27, 58, 107)"}}>Дигитална трансформација на јавните услуги</h2>
                    <p>е-Влада е официјалниот портал на Владата на Република Северна Македонија кој овозможува брз,
                        лесен и безбеден пристап до над 150 јавни услуги директно од дома.</p>
                    <p>Поднесување барања без чекање на ред</p>
                    <p>Проверка на статус на барање во реално време</p>
                    <p>Безбедно плаќање на такси и придонеси</p>
                    <p>АИ асистент за брза помош</p>
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

            {/* --- FOOTER --- */}
            <footer style={{background: '#0f2044', padding: '48px 60px 24px', fontFamily: "'Sora', sans-serif", marginTop: '60px'}}>
        <div style={{maxWidth: 1100, margin: '0 auto'}}>
          <div style={{display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: 48, paddingBottom: 40, borderBottom: '1px solid rgba(255,255,255,0.08)'}}>
            <div>
              <div style={{display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16}}>
                <div style={{width: 32, height: 32, borderRadius: 8, background: 'rgba(212,160,23,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16}}>🛡️</div>
                <span style={{color: '#D4A017', fontWeight: 700, fontSize: '1rem'}}>е-Влада</span>
              </div>
              <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', lineHeight: 1.7}}>Официјален портал на Владата на Република Северна Македонија за јавни услуги и информации.</p>
            </div>
            <div>
              <h4 style={{color: '#D4A017', fontSize: '0.82rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16}}>Брзи врски</h4>
              {[{name: 'Дома', path: '/'}, {name: 'ЧПП', path: '/faq'}, {name: 'Услуги', path: '/services'}, {name: 'Локации', path: '/locations'}].map(link => (
                <div key={link.name} style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '4px 0', cursor: 'pointer'}} onClick={() => navigate(link.path)}>{link.name}</div>
              ))}
            </div>
            <div>
              <h4 style={{color: '#D4A017', fontSize: '0.82rem', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16}}>Контакт</h4>
              <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>📞 +389 2 3145 100</p>
              <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>✉️ info@vlada.gov.mk</p>
              <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>📍 Илинденска б.б., Скопје</p>
            </div>
          </div>
          <div style={{textAlign: 'center', paddingTop: 24, color: 'rgba(255,255,255,0.25)', fontSize: '0.75rem'}}>
            © 2026 Влада на Република Северна Македонија. Сите права се задржани.
          </div>
        </div>
      </footer>

            {/* --- CHAT MODAL --- */}
            {showChat && (
                <div className="modal-overlay" onClick={() => setShowChat(false)}>
                    <div className="chat-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="chat-header">
                            <h3>АИ Асистент</h3>
                            <button className="close-btn" onClick={() => setShowChat(false)}>&times;</button>
                        </div>
                        <div className="chat-messages">
                            {chatMessages.map((msg, index) => (
                                <div key={index} className={`message ${msg.type}`}>
                                    {msg.type === "user" ? (
                                        <p><strong>Вие:</strong> {msg.text}</p>
                                    ) : msg.text ? (
                                        <p><strong>АИ:</strong> {msg.text}</p>
                                    ) : (
                                        <div className="faq-result">
                                            <p><strong>Прашање:</strong> {msg.question}</p>
                                            <p><strong>Одговор:</strong> {msg.answer}</p>
                                            <p><em>Довербa: {msg.confidence} ({msg.score.toFixed(3)})</em></p>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                        <form onSubmit={handleChatSubmit} className="chat-input-form">
                            <input
                                type="text"
                                value={chatInput}
                                onChange={(e) => setChatInput(e.target.value)}
                                placeholder="Поставете прашање..."
                                required
                            />
                            <button type="submit">Испрати</button>
                        </form>
                    </div>
                </div>
            )}
        </>
    );
}

export default Home;