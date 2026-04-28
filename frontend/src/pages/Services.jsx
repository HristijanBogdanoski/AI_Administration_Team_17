import React, {useEffect, useState} from "react";
import {useNavigate} from "react-router-dom";
import '../index.css';

const CATEGORY_TO_TAG = {
    documents: "Документи",
    taxes: "Даноци",
    social: "Социјални",
    business: "Деловни",
    education: "Услуги",
    utilities: "Институции"
};

const CATEGORY_TO_COLOR = {
    documents: "#1B3A6B",
    taxes: "#CE2028",
    social: "rgb(212,160,23)",
    business: "rgb(22,101,52)",
    education: "#8B5CF6",
    utilities: "rgb(3,105,161)"
};

const tags = ["Сите", "Документи", "Даноци", "Социјални", "Деловни", "Услуги", "Институции"];

function toDisplayTime(days) {
    if (days === null || days === undefined) return "Зависи";
    if (days === 0) return "Веднаш";
    return `${days} работни дена`;
}

export default function Services() {
    const navigate = useNavigate();

    // --- AUTHENTICATION STATE ---
    const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));
    const [showModal, setShowModal] = useState(false);
    const [authMode, setAuthMode] = useState("login");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [fullName, setFullName] = useState("");

    // --- UI & FILTER STATE ---
    const [selectedTag, setSelectedTag] = useState("Сите");
    const [search, setSearch] = useState("");
    const [selectedService, setSelectedService] = useState(null);
    const [services, setServices] = useState([]);
    const [loadingServices, setLoadingServices] = useState(true);
    const [selectedFormat, setSelectedFormat] = useState('txt');

    // --- AUTHENTICATION HANDLERS ---
    const handleAuth = async (e) => {
        e.preventDefault();
        const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
        const payload = authMode === "login"
            ? {username: email, password: password}
            : {email, full_name: fullName, password};

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
                    setIsLoggedIn(true);
                    setShowModal(false);
                    window.location.reload(); // Sync navbar on home/other pages
                } else {
                    alert("Успешна регистрација!");
                    setAuthMode("login");
                }
            } else {
                alert(data.detail || "Грешка при автентикација");
            }
        } catch (err) {
            console.error("Connection failed:", err);
        }
    };

    useEffect(() => {
        const fetchServices = async () => {
            try {
                setLoadingServices(true);
                const res = await fetch("http://127.0.0.1:8000/services");
                if (!res.ok) {
                    throw new Error("Неуспешно вчитување на услуги");
                }

                const data = await res.json();
                const mapped = data.map((service) => ({
                    id: service.id,
                    service_id: service.service_id,
                    name: service.name,
                    desc: service.description || "Нема опис за оваа услуга.",
                    tag: CATEGORY_TO_TAG[service.category] || "Услуги",
                    color: CATEGORY_TO_COLOR[service.category] || "#1B3A6B",
                    time: toDisplayTime(service.processing_time_days),
                    details: Array.isArray(service.details) ? service.details : [],
                    location: service.location,
                }));

                setServices(mapped);
            } catch (err) {
                console.error("Services fetch failed:", err);
                setServices([]);
            } finally {
                setLoadingServices(false);
            }
        };

        fetchServices();
    }, []);

    const filtered = services.filter((s) => {
        const matchTag = selectedTag === "Сите" || s.tag === selectedTag;
        const matchSearch = s.name.toLowerCase().includes(search.toLowerCase());
        return matchTag && matchSearch;
    });

    const downloadBlob = (blob, filename) => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
    };

    const handleDownloadDocument = async () => {
        if (!selectedService?.service_id) return;

        try {
            const response = await fetch(`http://127.0.0.1:8000/service-document-templates/${selectedService.service_id}/download?format=${encodeURIComponent(selectedFormat)}`);
            if (!response.ok) {
                alert("Неуспешно преземање на документот.");
                return;
            }

            const blob = await response.blob();
            const ext = selectedFormat === 'pdf' ? 'pdf' : selectedFormat === 'docx' ? 'docx' : 'txt';
            downloadBlob(blob, `${selectedService.service_id}-application-form.${ext}`);
        } catch {
            alert("Грешка при преземање на документот.");
        }
    };

    useEffect(() => {
        if (!selectedService && filtered.length > 0) {
            setSelectedService(filtered[0]);
        }
        if (selectedService && filtered.length > 0 && !filtered.some((s) => s.id === selectedService.id)) {
            setSelectedService(filtered[0]);
        }
        if (filtered.length === 0) {
            setSelectedService(null);
        }
    }, [filtered, selectedService]);

    return (
        <div style={{backgroundColor: "#f8fafc", minHeight: "100vh"}}>
            {/* NAVBAR REMOVED - Managed by App.js wrapper */}

            {/* --- AUTH MODAL --- */}
            {showModal && (
                <div style={{
                    position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: "rgba(0,0,0,0.6)", display: "flex",
                    justifyContent: "center", alignItems: "center", zIndex: 1000
                }} onClick={() => setShowModal(false)}>
                    <div style={{
                        backgroundColor: "#fff", padding: "40px", borderRadius: "12px",
                        width: "100%", maxWidth: "400px", position: "relative"
                    }} onClick={(e) => e.stopPropagation()}>
                        <button
                            style={{
                                position: "absolute",
                                top: "15px",
                                right: "15px",
                                border: "none",
                                background: "none",
                                fontSize: "24px",
                                cursor: "pointer"
                            }}
                            onClick={() => setShowModal(false)}
                        >&times;</button>
                        <div style={{display: "flex", marginBottom: "25px", borderBottom: "1px solid #eee"}}>
                            <button
                                style={{
                                    flex: 1,
                                    padding: "10px",
                                    border: "none",
                                    background: "none",
                                    cursor: "pointer",
                                    borderBottom: authMode === "login" ? "2px solid #1B3A6B" : "none",
                                    fontWeight: authMode === "login" ? "bold" : "normal"
                                }}
                                onClick={() => setAuthMode("login")}
                            >Најава
                            </button>
                            <button
                                style={{
                                    flex: 1,
                                    padding: "10px",
                                    border: "none",
                                    background: "none",
                                    cursor: "pointer",
                                    borderBottom: authMode === "register" ? "2px solid #1B3A6B" : "none",
                                    fontWeight: authMode === "register" ? "bold" : "normal"
                                }}
                                onClick={() => setAuthMode("register")}
                            >Регистрација
                            </button>
                        </div>
                        <form onSubmit={handleAuth} style={{display: "flex", flexDirection: "column", gap: "15px"}}>
                            {authMode === "register" && (
                                <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}}
                                       type="text" placeholder="Целосно име" value={fullName}
                                       onChange={(e) => setFullName(e.target.value)} required/>
                            )}
                            <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}} type="email"
                                   placeholder="Е-пошта" value={email} onChange={(e) => setEmail(e.target.value)}
                                   required/>
                            <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}}
                                   type="password" placeholder="Лозинка" value={password}
                                   onChange={(e) => setPassword(e.target.value)} required/>
                            <button type="submit" style={{
                                padding: "14px",
                                backgroundColor: "#1B3A6B",
                                color: "#fff",
                                border: "none",
                                borderRadius: "6px",
                                cursor: "pointer",
                                fontWeight: "bold"
                            }}>
                                {authMode === "login" ? "Влези" : "Креирај профил"}
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* --- HERO SECTION --- */}
            {/* --- HERO SECTION (MATCHED TO FAQ STYLE) --- */}
            {/* --- HERO SECTION (EXACT FAQ MATCH) --- */}
            <div style={{
                background: 'linear-gradient(150deg, #0f2044 0%, #1B3A6B 55%, #1a4a8a 100%)',
                // Matches the FAQ's exact padding: Top 52px, Sides 40px, Bottom 68px
                padding: '52px 40px 68px',
                textAlign: 'center',
                position: 'relative',
                overflow: 'hidden',
                fontFamily: "'Sora', sans-serif"
            }}>
                {/* The gold glow effect from the FAQ page */}
                <div style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'radial-gradient(ellipse 60% 70% at 50% 120%, rgba(212,160,23,0.1) 0%, transparent 70%)'
                }}/>

                {/* Icon box with SVG - exact size/margin match */}
                <div style={{
                    width: 64, height: 64,
                    background: 'rgba(255,255,255,0.1)',
                    backdropFilter: 'blur(12px)',
                    borderRadius: 18,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 22px',
                    border: '1px solid rgba(255,255,255,0.15)',
                    position: 'relative',
                    zIndex: 2
                }}>
                    {/* Replaced emoji with your Figma SVG */}
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="28"
                        height="28"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="rgb(212, 160, 23)" // Your Figma Gold
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <path
                            d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"></path>
                        <path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"></path>
                        <path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"></path>
                    </svg>
                </div>

                <h1 style={{
                    color: '#fff',
                    fontSize: '2.2rem',
                    fontWeight: 800,
                    letterSpacing: '-0.02em',
                    margin: '0 0 10px',
                    position: 'relative',
                    zIndex: 2
                }}>
                    Јавни Услуги
                </h1>

                {/* Added a subheader to fill the space like the FAQ page does */}
                <p style={{
                    color: '#93c5fd',
                    fontSize: '0.92rem',
                    margin: 0,
                    position: 'relative',
                    zIndex: 2
                }}>
                    Сите валидни услуги достапни на едно место
                </p>
            </div>

            {/* --- SEARCH & FILTERS --- */}
            {/* Changed margin from -40px to 40px to remove the overlap/collision */}
            <div style={{maxWidth: "1400px", margin: "40px auto", padding: "0 20px", position: "relative", zIndex: 10}}>
                <div style={{
                    backgroundColor: "transparent",
                    padding: "30px",
                    borderRadius: "16px",
                }}>
                    <div style={{display: "flex", justifyContent: "center", marginBottom: "30px"}}>
                        <div style={{position: "relative", width: "100%", maxWidth: "500px"}}>
                            <input
                                type="text"
                                placeholder="Пребарајте конкретна услуга (пр. Пасош)..."
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                style={{
                                    width: "100%",
                                    padding: "16px 20px 16px 50px",
                                    borderRadius: "30px",
                                    border: "1px solid #e2e8f0",
                                    fontSize: "1rem",
                                    outline: "none",
                                    boxSizing: "border-box" // Prevents input from expanding past container
                                }}
                            />
                            <span
                                style={{position: "absolute", left: "20px", top: "50%", transform: "translateY(-50%)"}}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2"><circle
                            cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                    </span>
                        </div>
                    </div>

                    <div style={{display: "flex", justifyContent: "center", gap: "12px", flexWrap: "wrap"}}>
                        {tags.map((tag) => (
                            <button
                                key={tag}
                                onClick={() => setSelectedTag(tag)}
                                style={{
                                    padding: "10px 24px", borderRadius: "25px", border: "1px solid",
                                    borderColor: selectedTag === tag ? "#1B3A6B" : "#e2e8f0",
                                    backgroundColor: selectedTag === tag ? "#1B3A6B" : "#fff",
                                    color: selectedTag === tag ? "#fff" : "#64748b",
                                    fontWeight: "600", cursor: "pointer",
                                    transition: "all 0.2s ease"
                                }}
                            >
                                {tag}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* --- CONTENT GRID --- */}
            <div style={{
                maxWidth: "1400px",
                margin: "0 auto 80px auto",
                padding: "0 20px",
                display: "grid",
                gridTemplateColumns: "1fr 450px",
                gap: "40px"
            }}>

                {/* LEFT COLUMN: LIST */}
                <div>
                    <h3 style={{marginBottom: "20px", color: "#1e293b"}}>
                        {loadingServices ? "Се вчитуваат услуги..." : `${filtered.length} Пронајдени услуги`}
                    </h3>
                    <div style={{display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "20px"}}>
                        {filtered.map((s, i) => (
                            <div
                                key={i}
                                onClick={() => setSelectedService(s)}
                                style={{
                                    backgroundColor: "#fff",
                                    padding: "24px",
                                    borderRadius: "15px",
                                    border: "1px solid #e2e8f0",
                                    cursor: "pointer",
                                    transition: "all 0.3s ease",
                                    boxShadow: selectedService?.name === s.name ? `0 0 0 2px ${s.color}` : "none",
                                    transform: selectedService?.name === s.name ? "scale(1.02)" : "scale(1)"
                                }}
                            >
                                <div style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    alignItems: "flex-start",
                                    marginBottom: "12px"
                                }}>
                                    <h4 style={{margin: 0, fontSize: "1.1rem", color: "#1e293b"}}>{s.name}</h4>
                                    <span style={{
                                        fontSize: "0.7rem",
                                        padding: "4px 8px",
                                        borderRadius: "4px",
                                        backgroundColor: `${s.color}15`,
                                        color: s.color,
                                        fontWeight: "700"
                                    }}>{s.tag}</span>
                                </div>
                                <p style={{fontSize: "0.9rem", color: "#64748b", marginBottom: "15px"}}>{s.desc}</p>
                                <div style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "6px",
                                    fontSize: "0.8rem",
                                    color: "#94a3b8"
                                }}>
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                                         strokeWidth="2">
                                        <circle cx="12" cy="12" r="10"/>
                                        <path d="M12 6v6l4 2"/>
                                    </svg>
                                    {s.time}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* RIGHT COLUMN: DETAILS PANEL */}
                <div style={{position: "sticky", top: "20px", height: "fit-content"}}>
                    <div style={{
                        backgroundColor: "#fff",
                        padding: "40px",
                        borderRadius: "20px",
                        border: "1px solid #e2e8f0",
                        minHeight: "500px",
                        display: "flex",
                        flexDirection: "column",
                        gap: 14 // Consistent spacing between elements
                    }}>
                        {selectedService ? (
                            <>
                                {/* Name and Tag in one line */}
                                <div style={{display: "flex", alignItems: "center", gap: "12px"}}>
                                    <h2 style={{margin: 0, fontSize: "2rem", color: "#1e293b", fontWeight: 700}}>
                                        {selectedService.name}
                                    </h2>
                                    <span style={{
                                        fontSize: "0.78rem",
                                        color: selectedService.color,
                                        background: `${selectedService.color}15`,
                                        padding: "4px 10px",
                                        borderRadius: 6,
                                        whiteSpace: "nowrap",
                                        fontWeight: 700
                                    }}>
                        {selectedService.tag}
                    </span>
                                </div>

                                <p style={{fontSize: "1.05rem", color: "#475569", lineHeight: "1.7", margin: "10px 0"}}>
                                    {selectedService.desc}
                                </p>

                                <div>
                                    <h4 style={{marginBottom: 12, color: "#1e293b"}}>Детали</h4>
                                    {selectedService.details?.map((d, i) => (
                                        <div key={i} style={{color: "#64748b", padding: "4px 0", fontSize: "0.95rem"}}>
                                            • {d}
                                        </div>
                                    ))}
                                </div>

                                <div>
                                    <h4 style={{marginBottom: 4, color: "#1e293b"}}>Рок на обработка</h4>
                                    <p style={{color: "#94a3b8", fontWeight: 500, margin: 0}}>{selectedService.time}</p>
                                </div>

                                {/* TWO BUTTONS AT THE BOTTOM */}
                                <div style={{marginTop: "auto", display: "flex", flexDirection: "column", gap: 12}}>
                                    <div style={{display: 'flex', gap: 8, marginBottom: 8}}>
                                        <select value={selectedFormat} onChange={(e) => setSelectedFormat(e.target.value)} style={{padding: '8px', borderRadius: 8, border: '1px solid #e2e8f0'}}>
                                            <option value="txt">TXT</option>
                                            <option value="pdf">PDF</option>
                                            <option value="docx">Word (.docx)</option>
                                        </select>
                                    </div>

                                    <button
                                        onClick={handleDownloadDocument}
                                        style={{
                                            background: selectedService.color,
                                            color: "#fff",
                                            border: "none",
                                            padding: "14px",
                                            borderRadius: "10px",
                                            cursor: "pointer",
                                            fontWeight: "600",
                                            fontSize: "1rem",
                                            transition: "opacity 0.2s ease",
                                        }}
                                        onMouseOver={(e) => e.target.style.opacity = "0.85"}
                                        onMouseOut={(e) => e.target.style.opacity = "1"}
                                    >
                                        Преземи документ
                                    </button>
                                    <p style={{margin: 0, fontSize: "0.86rem", color: "#64748b", lineHeight: 1.55}}>
                                        Документот можете да го пополните и автоматски во АИ Чет. Таму изберете „Прикачи документ за пополнување“.
                                    </p>
                                    {/* Secondary Button - Location */}
                                    <button
                                        onClick={async () => {
                                            navigate('/locations', {
                                                state: {
                                                    serviceId: selectedService.service_id || null,
                                                    serviceName: selectedService.name || null,
                                                },
                                            });
                                        }}
                                        style={{
                                            background: "transparent",
                                            border: "1px solid #e2e8f0",
                                            color: "#64748b",
                                            padding: "14px",
                                            borderRadius: "10px",
                                            cursor: "pointer",
                                            fontWeight: "600",
                                            fontSize: "1rem"
                                        }}
                                    >
                                        Локација
                                    </button>
                                </div>
                            </>
                        ) : (
                            <div style={{
                                flex: 1,
                                display: "flex",
                                flexDirection: "column",
                                justifyContent: "center",
                                alignItems: "center",
                                textAlign: "center",
                                color: "#94a3b8"
                            }}>
                                <h3 style={{fontWeight: 500}}>Изберете услуга</h3>
                                <p>Кликнете на некоја од услугите лево за да ги видите деталите.</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* --- EXACT FOOTER FROM FAQ PAGE --- */}
            <footer style={{background: '#0f2044', padding: '48px 60px 24px', fontFamily: "'Sora', sans-serif"}}>
                <div style={{maxWidth: 1100, margin: '0 auto'}}>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '2fr 1fr 1fr',
                        gap: 48,
                        paddingBottom: 40,
                        borderBottom: '1px solid rgba(255,255,255,0.08)'
                    }}>
                        <div>
                            <div style={{display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16}}>
                                <div style={{
                                    width: 32,
                                    height: 32,
                                    borderRadius: 8,
                                    background: 'rgba(212,160,23,0.2)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: 16
                                }}>🛡️
                                </div>
                                <span style={{color: '#D4A017', fontWeight: 700, fontSize: '1rem'}}>е-Влада</span>
                            </div>
                            <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', lineHeight: 1.7}}>
                                Официјален портал на Владата на Република Северна Македонија за јавни услуги и
                                информации.
                            </p>
                        </div>
                        <div>
                            <h4 style={{
                                color: '#D4A017',
                                fontSize: '0.82rem',
                                fontWeight: 700,
                                letterSpacing: '0.08em',
                                textTransform: 'uppercase',
                                marginBottom: 16
                            }}>Брзи врски</h4>
                            {[
                                {name: 'Дома', path: '/'},
                                {name: 'ЧПП', path: '/faq'},
                                {name: 'Услуги', path: '/services'},
                                {name: 'Локација', path: '/'}
                            ].map(link => (
                                <div
                                    key={link.name}
                                    style={{
                                        color: 'rgba(255,255,255,0.5)',
                                        fontSize: '0.82rem',
                                        padding: '4px 0',
                                        cursor: 'pointer'
                                    }}
                                    onClick={() => navigate(link.path)}
                                >
                                    {link.name}
                                </div>
                            ))}
                        </div>
                        <div>
                            <h4 style={{
                                color: '#D4A017',
                                fontSize: '0.82rem',
                                fontWeight: 700,
                                letterSpacing: '0.08em',
                                textTransform: 'uppercase',
                                marginBottom: 16
                            }}>Контакт</h4>
                            <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>📞 +389 2
                                3145 100</p>
                            <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>✉️
                                info@vlada.gov.mk</p>
                            <p style={{color: 'rgba(255,255,255,0.5)', fontSize: '0.82rem', padding: '3px 0'}}>📍
                                Илинденска б.б., Скопје</p>
                        </div>
                    </div>
                    <div style={{
                        textAlign: 'center',
                        paddingTop: 24,
                        color: 'rgba(255,255,255,0.25)',
                        fontSize: '0.75rem'
                    }}>
                        © 2026 Влада на Република Северна Македонија. Сите права се задржани.
                    </div>
                </div>
            </footer>
        </div>
    );
}