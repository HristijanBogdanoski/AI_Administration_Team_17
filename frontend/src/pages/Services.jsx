import React, {useEffect, useState} from "react";
import {useNavigate} from "react-router-dom";
import '../index.css';

const CATEGORY_TO_TAG = {documents: "Документи", taxes: "Даноци", social: "Социјални", business: "Деловни", education: "Услуги", utilities: "Институции"};
const CATEGORY_TO_COLOR = {documents: "#1B3A6B", taxes: "#CE2028", social: "rgb(212,160,23)", business: "rgb(22,101,52)", education: "#8B5CF6", utilities: "rgb(3,105,161)"};
const TAGS = ["Сите", "Документи", "Даноци", "Социјални", "Деловни", "Услуги", "Институции"];

const toDisplayTime = (days) => {
    if (days === null || days === undefined) return "Зависи";
    if (days === 0) return "Веднаш";
    return `${days} работни дена`;
};

export default function Services() {
    const navigate = useNavigate();

    // --- AUTHENTICATION STATE ---
    const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));
    const [showModal, setShowModal] = useState(false);
    const [authMode, setAuthMode] = useState("login");
    const [authForm, setAuthForm] = useState({email: "", password: "", fullName: ""});
    const [selectedTag, setSelectedTag] = useState("Сите");
    const [search, setSearch] = useState("");
    const [selectedService, setSelectedService] = useState(null);
    const [services, setServices] = useState([]);
    const [loadingServices, setLoadingServices] = useState(true);
    const [selectedFormat, setSelectedFormat] = useState('txt');
    const [showCrudModal, setShowCrudModal] = useState(false);
    const [modalMode, setModalMode] = useState('create');
    const [selectedServiceForCrud, setSelectedServiceForCrud] = useState(null);
    const [crudError, setCrudError] = useState('');
    const [formData, setFormData] = useState({service_id: '', name: '', category: 'documents', description: '', processing_time_days: 0, location: ''});
    const [showTemplateModal, setShowTemplateModal] = useState(false);
    const [templateMode, setTemplateMode] = useState('create');
    const [templateData, setTemplateData] = useState({service_id: '', title: '', template_body: '', is_active: true});
    const [currentTemplate, setCurrentTemplate] = useState(null);

    // --- TEMPLATE HANDLERS ---
    const handleTemplateCrud = (mode) => {
        setTemplateMode(mode);
        if (mode === 'create') {
            setTemplateData({service_id: selectedService?.service_id || '', title: '', template_body: '', is_active: true});
        } else if (mode === 'edit' && currentTemplate) {
            setTemplateData({
                service_id: currentTemplate.service_id,
                title: currentTemplate.title,
                template_body: JSON.stringify(currentTemplate.template_body, null, 2),
                is_active: currentTemplate.is_active
            });
        }
        setShowTemplateModal(true);
    };

    const submitTemplate = async () => {
        const token = localStorage.getItem("token");
        let url = "http://127.0.0.1:8000/service-document-templates";
        let method = "POST";
        
        if (templateMode === 'edit') {
            url += `/${templateData.service_id}`;
            method = "PUT";
        } else if (templateMode === 'delete') {
            url += `/${currentTemplate.service_id}`;
            method = "DELETE";
        }
        
        try {
            const response = await fetch(url, {
                method,
                headers: {"Content-Type": "application/json", "Authorization": `Bearer ${token}`},
                body: templateMode !== 'delete' ? JSON.stringify({...templateData, template_body: JSON.parse(templateData.template_body)}) : undefined
            });
            if (response.ok) {
                setShowTemplateModal(false);
                if (templateMode === 'delete') {
                    setCurrentTemplate(null);
                } else {
                    fetchTemplateForService(selectedService.service_id);
                }
            }
        } catch (err) {
            console.error('Template error:', err);
        }
    };

    const fetchTemplateForService = async (serviceId) => {
        const token = localStorage.getItem("token");
        try {
            const response = await fetch(`http://127.0.0.1:8000/service-document-templates/${serviceId}`, {
                headers: {"Authorization": `Bearer ${token}`}
            });
            if (response.ok) {
                const template = await response.json();
                setCurrentTemplate(template);
            } else {
                setCurrentTemplate(null);
            }
        } catch (err) {
            setCurrentTemplate(null);
        }
    };

    const handleCrud = async (mode, service = null) => {
        setModalMode(mode);
        setSelectedServiceForCrud(service);
        setCrudError('');
        if (mode === 'create') {
            setFormData({service_id: '', name: '', category: 'documents', description: '', processing_time_days: 0, location: '', details: []});
        } else if (mode === 'edit' && service) {
            const processingDays = service.time === 'Веднаш' ? 0 : (typeof service.time === 'number' ? service.time : parseInt(service.time) || 0);
            setFormData({
                service_id: service.service_id,
                name: service.name,
                category: Object.keys(CATEGORY_TO_TAG).find(k => CATEGORY_TO_TAG[k] === service.tag) || 'documents',
                description: service.desc,
                processing_time_days: processingDays,
                location: service.location || '',
                details: service.details || []
            });
        }
        setShowCrudModal(true);
    };

    const submitCrud = async () => {
        const token = localStorage.getItem("token");
        let url = "http://127.0.0.1:8000/services";
        let method = "POST";
        
        if (modalMode === 'edit') {
            url += `/${selectedServiceForCrud.id}`;
            method = "PUT";
        } else if (modalMode === 'delete') {
            url += `/${selectedServiceForCrud.id}`;
            method = "DELETE";
        }

        if (modalMode === 'create') {
            const existingService = services.find(s => s.service_id === formData.service_id);
            if (existingService) {
                setCrudError(`Услуга со ID "${formData.service_id}" веќе постои ("${existingService.name}"). Изберете друг ID.`);
                return;
            }
        }

        const dataToSend = modalMode !== 'delete' ? {...formData} : undefined;

        try {
            const response = await fetch(url, {
                method,
                headers: {"Content-Type": "application/json", "Authorization": `Bearer ${token}`},
                body: modalMode !== 'delete' ? JSON.stringify(dataToSend) : undefined
            });
            
            if (response.ok) {
                if (modalMode === 'delete') {
                    setServices(prev => prev.filter(s => s.id !== selectedServiceForCrud.id));
                } else {
                    const result = await response.json();
                    if (modalMode === 'create') {
                        setServices(prev => [{...result, tag: CATEGORY_TO_TAG[result.category] || "Услуги", color: CATEGORY_TO_COLOR[result.category] || "#1B3A6B", time: toDisplayTime(result.processing_time_days)}, ...prev]);
                    } else {
                        setServices(prev => prev.map(s => {
                            if (s.id === selectedServiceForCrud.id) {
                                return {
                                    ...s,
                                    ...result,
                                    desc: result.description || s.desc,
                                    tag: CATEGORY_TO_TAG[result.category] || CATEGORY_TO_TAG[s.category] || "Услуги",
                                    color: CATEGORY_TO_COLOR[result.category] || CATEGORY_TO_COLOR[s.category] || "#1B3A6B",
                                    time: toDisplayTime(result.processing_time_days !== undefined ? result.processing_time_days : s.processing_time_days)
                                };
                            }
                            return s;
                        }));
                    }
                }
                setShowCrudModal(false);
            } else {
                const error = await response.json();
                setCrudError(error.detail || 'Грешка при зачувување. Обидете се повторно.');
            }
        } catch (err) {
            console.error('Operation error:', err);
        }
    };

    // --- AUTHENTICATION HANDLERS ---
    const handleAuth = async (e) => {
        e.preventDefault();
        const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
        const payload = authMode === "login" ? {username: authForm.email, password: authForm.password} : {email: authForm.email, full_name: authForm.fullName, password: authForm.password};

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
                    window.location.reload();
                } else {
                    setAuthMode("login");
                }
            } else {
                console.error(data.detail || "Грешка при автентикација");
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
                if (!res.ok) throw new Error("Неуспешно вчитување на услуги");
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
                const sortedMapped = mapped.sort((a, b) => (a.id && b.id) ? b.id - a.id : 0);
                setServices(sortedMapped);
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
                console.error("Неуспешно преземање на документот.");
                return;
            }
            const blob = await response.blob();
            const ext = selectedFormat === 'pdf' ? 'pdf' : selectedFormat === 'docx' ? 'docx' : 'txt';
            downloadBlob(blob, `${selectedService.service_id}-application-form.${ext}`);
        } catch {
            console.error("Грешка при преземање на документот.");
        }
    };

    useEffect(() => {
        if (!selectedService && filtered.length > 0) setSelectedService(filtered[0]);
        if (selectedService && filtered.length > 0 && !filtered.some((s) => s.id === selectedService.id)) setSelectedService(filtered[0]);
        if (filtered.length === 0) setSelectedService(null);
    }, [filtered, selectedService]);

    useEffect(() => {
        if (selectedService && isLoggedIn) {
            fetchTemplateForService(selectedService.service_id);
        }
    }, [selectedService, isLoggedIn]);

    return (
        <div style={{backgroundColor: "#f8fafc", minHeight: "100vh"}}>
            {/* NAVBAR REMOVED - Managed by App.js wrapper */}

            {/* --- AUTH MODAL --- */}
            {showModal && (
                <div style={{position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000}} onClick={() => setShowModal(false)}>
                    <div style={{backgroundColor: "#fff", padding: "40px", borderRadius: "12px", width: "100%", maxWidth: "400px", position: "relative"}} onClick={(e) => e.stopPropagation()}>
                        <button style={{position: "absolute", top: "15px", right: "15px", border: "none", background: "none", fontSize: "24px", cursor: "pointer"}} onClick={() => setShowModal(false)}>&times;</button>
                        <div style={{display: "flex", marginBottom: "25px", borderBottom: "1px solid #eee"}}>
                            <button style={{flex: 1, padding: "10px", border: "none", background: "none", cursor: "pointer", borderBottom: authMode === "login" ? "2px solid #1B3A6B" : "none", fontWeight: authMode === "login" ? "bold" : "normal"}} onClick={() => setAuthMode("login")}>Најава</button>
                            <button style={{flex: 1, padding: "10px", border: "none", background: "none", cursor: "pointer", borderBottom: authMode === "register" ? "2px solid #1B3A6B" : "none", fontWeight: authMode === "register" ? "bold" : "normal"}} onClick={() => setAuthMode("register")}>Регистрација</button>
                        </div>
                        <form onSubmit={handleAuth} style={{display: "flex", flexDirection: "column", gap: "15px"}}>
                            {authMode === "register" && <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}} type="text" placeholder="Целосно име" value={authForm.fullName} onChange={(e) => setAuthForm({...authForm, fullName: e.target.value})} required/>}
                            <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}} type="email" placeholder="Е-пошта" value={authForm.email} onChange={(e) => setAuthForm({...authForm, email: e.target.value})} required/>
                            <input style={{padding: "12px", borderRadius: "6px", border: "1px solid #ddd"}} type="password" placeholder="Лозинка" value={authForm.password} onChange={(e) => setAuthForm({...authForm, password: e.target.value})} required/>
                            <button type="submit" style={{padding: "14px", backgroundColor: "#1B3A6B", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer", fontWeight: "bold"}}>
                                {authMode === "login" ? "Влези" : "Креирај профил"}
                            </button>
                        </form>
                    </div>
                </div>
            )}

            <div style={{background: 'linear-gradient(150deg, #0f2044 0%, #1B3A6B 55%, #1a4a8a 100%)', padding: '52px 40px 68px', textAlign: 'center', position: 'relative', overflow: 'hidden', fontFamily: "'Sora', sans-serif"}}>
                <div style={{position: 'absolute', inset: 0, background: 'radial-gradient(ellipse 60% 70% at 50% 120%, rgba(212,160,23,0.1) 0%, transparent 70%)'}}/>
                <div style={{width: 64, height: 64, background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(12px)', borderRadius: 18, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 22px', border: '1px solid rgba(255,255,255,0.15)', position: 'relative', zIndex: 2}}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="rgb(212, 160, 23)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"></path>
                        <path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"></path>
                        <path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"></path>
                    </svg>
                </div>
                <h1 style={{color: '#fff', fontSize: '2.2rem', fontWeight: 800, letterSpacing: '-0.02em', margin: '0 0 10px', position: 'relative', zIndex: 2}}>Јавни Услуги</h1>
                <p style={{color: '#93c5fd', fontSize: '0.92rem', margin: 0, position: 'relative', zIndex: 2}}>Сите валидни услуги достапни на едно место</p>
            </div>

            <div style={{maxWidth: "1400px", margin: "40px auto", padding: "0 20px", position: "relative", zIndex: 10}}>
                <div style={{backgroundColor: "transparent", padding: "30px", borderRadius: "16px"}}>
                    <div style={{display: "flex", justifyContent: "center", marginBottom: "30px"}}>
                        <div style={{position: "relative", width: "100%", maxWidth: "500px"}}>
                            <input type="text" placeholder="Пребарајте конкретна услуга (пр. Пасош)..." value={search} onChange={(e) => setSearch(e.target.value)} style={{width: "100%", padding: "16px 20px 16px 50px", borderRadius: "30px", border: "1px solid #e2e8f0", fontSize: "1rem", outline: "none", boxSizing: "border-box"}}/>
                            <span style={{position: "absolute", left: "20px", top: "50%", transform: "translateY(-50%)"}}>
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                            </span>
                        </div>
                    </div>

                    <div style={{display: "flex", justifyContent: "center", gap: "12px", flexWrap: "wrap"}}>
                        {TAGS.map((tag) => (
                            <button key={tag} onClick={() => setSelectedTag(tag)} style={{padding: "10px 24px", borderRadius: "25px", border: "1px solid", borderColor: selectedTag === tag ? "#1B3A6B" : "#e2e8f0", backgroundColor: selectedTag === tag ? "#1B3A6B" : "#fff", color: selectedTag === tag ? "#fff" : "#64748b", fontWeight: "600", cursor: "pointer", transition: "all 0.2s ease"}}>{tag}</button>
                        ))}
                    </div>

                    {isLoggedIn && (
                        <div style={{display: "flex", justifyContent: "center", marginTop: "20px"}}>
                            <button onClick={() => handleCrud('create')} style={{backgroundColor: "#1B3A6B", color: "#fff", border: "none", padding: "12px 24px", borderRadius: "25px", cursor: "pointer", fontWeight: "600", fontSize: "0.9rem", transition: "all 0.2s ease"}}>+ Додади Услуга</button>
                        </div>
                    )}
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
                                    <div style={{display: "flex", gap: "5px", alignItems: "center"}}>
                                        <span style={{
                                            fontSize: "0.7rem",
                                            padding: "4px 8px",
                                            borderRadius: "4px",
                                            backgroundColor: `${s.color}15`,
                                            color: s.color,
                                            fontWeight: "700"
                                        }}>{s.tag}</span>
                                        {isLoggedIn && (
                                            <div style={{display: "flex", gap: "3px"}}>
                                                <button onClick={(e) => {e.stopPropagation(); handleCrud('edit', s);}} style={{backgroundColor: "#3b82f6", color: "#fff", border: "none", padding: "3px 6px", borderRadius: "3px", cursor: "pointer", fontSize: "0.7rem"}}>Измени</button>
                                                <button onClick={(e) => {e.stopPropagation(); handleCrud('delete', s);}} style={{backgroundColor: "#ef4444", color: "#fff", border: "none", padding: "3px 6px", borderRadius: "3px", cursor: "pointer", fontSize: "0.7rem"}}>Избриши</button>
                                            </div>
                                        )}
                                    </div>
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

                                {/* TEMPLATE ADMIN FOR LOGGED IN USERS */}
                                    {isLoggedIn && (
                                        <div style={{marginTop: 12, padding: "10px", backgroundColor: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0"}}>
                                            <div style={{display: "flex", alignItems: "center", gap: 8, marginBottom: 8}}>
                                                <div style={{width: 8, height: 8, borderRadius: "50%", backgroundColor: currentTemplate ? "#10b981" : "#ef4444"}}/>
                                                <span style={{fontSize: "0.85rem", color: "#64748b"}}>
                                                    {currentTemplate ? "Шаблон постои" : "Нема шаблон"}
                                                </span>
                                            </div>
                                            <div style={{display: "flex", gap: 6}}>
                                                {!currentTemplate && <button onClick={() => handleTemplateCrud('create')} style={{backgroundColor: "#10b981", color: "#fff", border: "none", padding: "4px 8px", borderRadius: "4px", cursor: "pointer", fontSize: "0.75rem"}}>+ Додади</button>}
                                                {currentTemplate && (
                                                    <>
                                                        <button onClick={() => handleTemplateCrud('edit')} style={{backgroundColor: "#3b82f6", color: "#fff", border: "none", padding: "4px 8px", borderRadius: "4px", cursor: "pointer", fontSize: "0.75rem"}}>Измени</button>
                                                        <button onClick={() => handleTemplateCrud('delete')} style={{backgroundColor: "#ef4444", color: "#fff", border: "none", padding: "4px 8px", borderRadius: "4px", cursor: "pointer", fontSize: "0.75rem"}}>Избриши</button>
                                                    </>
                                                )}
                                            </div>
                                        </div>
                                    )}

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

            {/* --- COMPACT CRUD MODAL --- */}
            {showCrudModal && (
                <div style={{position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000}} onClick={() => setShowCrudModal(false)}>
                    <div style={{backgroundColor: "#fff", padding: "30px", borderRadius: "12px", width: "100%", maxWidth: modalMode === 'delete' ? "400px" : "500px", position: "relative"}} onClick={(e) => e.stopPropagation()}>
                        <button style={{position: "absolute", top: "10px", right: "10px", border: "none", background: "none", fontSize: "20px", cursor: "pointer"}} onClick={() => setShowCrudModal(false)}>&times;</button>
                        
                        <h3 style={{marginBottom: "20px", color: modalMode === 'delete' ? "#dc2626" : "#1e293b"}}>
                            {modalMode === 'create' ? 'Нова Услуга' : modalMode === 'edit' ? 'Ажурирај Услуга' : 'Избриши Услуга'}
                        </h3>
                        
                        {modalMode === 'delete' ? (
                            <div>
                                <p style={{marginBottom: "20px", color: "#6b7280"}}>
                                    Дали сте сигурни дека сакате да ја избришете услугата "<strong>{selectedServiceForCrud?.name}</strong>" (ID: <strong>{selectedServiceForCrud?.service_id}</strong>)?
                                </p>
                                <div style={{display: "flex", gap: "10px"}}>
                                    <button onClick={submitCrud} style={{flex: 1, padding: "10px", backgroundColor: "#dc2626", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Избриши</button>
                                    <button onClick={() => setShowCrudModal(false)} style={{flex: 1, padding: "10px", backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Откажи</button>
                                </div>
                            </div>
                        ) : (
                            <form onSubmit={(e) => {e.preventDefault(); submitCrud();}} style={{display: "flex", flexDirection: "column", gap: "15px"}}>
                                {modalMode === 'edit' && (
                                    <div style={{marginBottom: "10px"}}>
                                        <small style={{color: "#6b7280", fontSize: "0.8rem", display: "block", marginBottom: "5px"}}>
                                            ID на Услуга (не може да се промени):
                                        </small>
                                        <input 
                                            type="text" 
                                            value={formData.service_id} 
                                            disabled
                                            style={{
                                                padding: "10px", 
                                                border: "1px solid #d1d5db", 
                                                borderRadius: "6px",
                                                backgroundColor: "#f3f4f6",
                                                color: "#6b7280",
                                                fontWeight: "bold"
                                            }} 
                                            readOnly
                                        />
                                    </div>
                                )}
                                {modalMode === 'create' && (
                                    <input 
                                        type="text" 
                                        placeholder="ID на Услуга *" 
                                        value={formData.service_id} 
                                        onChange={(e) => setFormData({...formData, service_id: e.target.value})}
                                        style={{
                                            padding: "10px", 
                                            border: "1px solid #d1d5db", 
                                            borderRadius: "6px"
                                        }} 
                                        required 
                                    />
                                )}
                                <input 
                                    type="text" 
                                    placeholder="Име на Услуга *" 
                                    value={formData.name} 
                                    onChange={(e) => setFormData({...formData, name: e.target.value})} 
                                    style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} 
                                    required 
                                />
                                {modalMode === 'create' && formData.service_id && (
                                    <div style={{fontSize: "0.75rem", color: "#6b7280", marginTop: "-10px", paddingLeft: "4px"}}>
                                        ID: <strong>{formData.service_id}</strong>
                                    </div>
                                )}
                                <select value={formData.category} onChange={(e) => setFormData({...formData, category: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}}>
                                    <option value="documents">Документи</option>
                                    <option value="taxes">Даноци</option>
                                    <option value="social">Социјални</option>
                                    <option value="business">Деловни</option>
                                    <option value="education">Услуги</option>
                                    <option value="utilities">Институции</option>
                                </select>
                                <textarea 
                                    placeholder="Опис" 
                                    value={formData.description || ''} 
                                    onChange={(e) => {
                                        console.log('Description changed from:', formData.description, 'to:', e.target.value);
                                        setFormData({...formData, description: e.target.value});
                                    }} 
                                    style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px", minHeight: "60px"}}
                                />
                                <textarea 
                                    placeholder="Детали (еден по ред)" 
                                    value={formData.details ? formData.details.join('\n') : ''} 
                                    onChange={(e) => setFormData({...formData, details: e.target.value.split('\n')})} 
                                    onBlur={(e) => setFormData({...formData, details: e.target.value.split('\n').filter(d => d.trim())})}
                                    style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px", minHeight: "80px"}}
                                />
                                <input 
    type="text" 
    placeholder={formData.processing_time_days === 0 ? "Работни Дена" : "Изберете работни денови (пр. 3 работни дена)"} 
    value={formData.processing_time_days === 0 ? "" : formData.processing_time_days} 
    onChange={(e) => setFormData({...formData, processing_time_days: e.target.value})} 
    style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} 
/>
                                {crudError && (
                                    <div style={{padding: "8px 12px", backgroundColor: "#fef2f2", border: "1px solid #fecaca", borderRadius: "6px", color: "#dc2626", fontSize: "0.85rem"}}>
                                        {crudError}
                                    </div>
                                )}
                                <div style={{display: "flex", gap: "10px"}}>
                                    <button type="submit" style={{flex: 1, padding: "10px", backgroundColor: "#1B3A6B", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>{modalMode === 'edit' ? 'Ажурирај' : 'Креирај'}</button>
                                    <button type="button" onClick={() => setShowCrudModal(false)} style={{flex: 1, padding: "10px", backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Откажи</button>
                                </div>
                            </form>
                        )}
                    </div>
                </div>
            )}

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

            {/* --- TEMPLATE MODAL --- */}
            {showTemplateModal && (
                <div style={{position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000}} onClick={() => setShowTemplateModal(false)}>
                    <div style={{backgroundColor: "#fff", padding: "30px", borderRadius: "12px", width: "100%", maxWidth: "500px", position: "relative"}} onClick={(e) => e.stopPropagation()}>
                        <button style={{position: "absolute", top: "10px", right: "10px", border: "none", background: "none", fontSize: "20px", cursor: "pointer"}} onClick={() => setShowTemplateModal(false)}>&times;</button>
                        <h3 style={{marginBottom: "20px", color: "#1e293b"}}>{templateMode === 'create' ? 'Нов Шаблон' : 'Измени Шаблон'}</h3>
                        <form onSubmit={(e) => {e.preventDefault(); submitTemplate();}} style={{display: "flex", flexDirection: "column", gap: "15px"}}>
                            <input type="text" placeholder="Наслов" value={templateData.title} onChange={(e) => setTemplateData({...templateData, title: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} required/>
                            <textarea placeholder="Template Body (JSON)" value={templateData.template_body} onChange={(e) => setTemplateData({...templateData, template_body: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px", minHeight: "100px"}} required/>
                            <div style={{display: "flex", gap: "10px"}}>
                                <button type="submit" style={{flex: 1, padding: "10px", backgroundColor: "#1B3A6B", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>{templateMode === 'create' ? 'Креирај' : 'Ажурирај'}</button>
                                <button type="button" onClick={() => setShowTemplateModal(false)} style={{flex: 1, padding: "10px", backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Откажи</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}