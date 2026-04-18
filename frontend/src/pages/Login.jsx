import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const roles = [
    {
        id: 'citizen',
        title: 'Граѓанин',
        desc: 'Пристап до лични документи, даноци и социјални услуги',
        icon: '👤',
        color: '#1B3A6B',
        bg: '#EFF6FF',
        features: ['Лични документи', 'Даночна пријава', 'Социјална помош', 'Закажување термини'],
    },
    {
        id: 'admin',
        title: 'Администратор',
        desc: 'Целосен административен пристап до системот',
        icon: '🔑',
        color: '#CE2028',
        bg: '#FFF1F2',
        features: ['Управување со корисници', 'Конфигурација', 'Системски логови', 'Безбедност'],
    },
];

function Login() {
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState('login');
    const [showPassword, setShowPassword] = useState(false);
    const [selectedRole, setSelectedRole] = useState('citizen');
    const [loginForm, setLoginForm] = useState({ email: '', password: '' });
    const [registerForm, setRegisterForm] = useState({ name: '', email: '', password: '', confirm: '', embg: '' });
    const [loginSuccess, setLoginSuccess] = useState(false);
    const [registerSuccess, setRegisterSuccess] = useState(false);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const switchTab = (tab) => {
        setActiveTab(tab);
        setError('');
        setLoginSuccess(false);
        setRegisterSuccess(false);
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const response = await fetch('http://127.0.0.1:8000/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: loginForm.email, password: loginForm.password }),
            });
            const data = await response.json();
            if (response.ok) {
                localStorage.setItem('token', data.access_token);
                setLoginSuccess(true);
                setTimeout(() => navigate('/'), 1500);
            } else {
                setError(data.detail || 'Погрешна е-пошта или лозинка.');
            }
        } catch {
            setError('Грешка при поврзување со серверот.');
        } finally {
            setLoading(false);
        }
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setError('');
        if (registerForm.password !== registerForm.confirm) {
            setError('Лозинките не се совпаѓаат.');
            return;
        }
        setLoading(true);
        try {
            const response = await fetch('http://127.0.0.1:8000/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: registerForm.email, full_name: registerForm.name, password: registerForm.password }),
            });
            const data = await response.json();
            if (response.ok) {
                setRegisterSuccess(true);
            } else {
                setError(data.detail || 'Грешка при регистрација.');
            }
        } catch {
            setError('Грешка при поврзување со серверот.');
        } finally {
            setLoading(false);
        }
    };

    const tabs = [
        { id: 'login', label: 'Најава' },
        { id: 'register', label: 'Регистрација' },
        { id: 'roles', label: 'Улоги' },
    ];

    return (
        <div style={{ minHeight: '100vh', background: '#F4F6FA' }}>

            {/* Red Banner */}
            <div style={{ background: '#CE2028', padding: '10px 16px', textAlign: 'center' }}>
                <p style={{ color: '#fff', margin: 0, fontSize: '0.875rem', fontWeight: 500 }}>
                    Портал на Владата на Република Северна Македонија
                </p>
            </div>

            {/* Navbar */}
            <div style={{
                background: '#1B3A6B', padding: '0 32px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                height: 64
            }}>
                {/* Logo */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}
                     onClick={() => navigate('/')}>
                    <div style={{
                        width: 44, height: 44, borderRadius: '50%',
                        background: 'rgba(212,160,23,0.2)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22
                    }}>🛡️</div>
                    <div>
                        <p style={{ color: '#D4A017', fontWeight: 700, fontSize: '1rem', margin: 0, lineHeight: 1.2 }}>
                            е-Влада
                        </p>
                        <p style={{ color: '#93c5fd', fontSize: '0.7rem', margin: 0 }}>Јавни Услуги</p>
                    </div>
                </div>

                {/* Nav Links */}
                <div style={{ display: 'flex', gap: 4 }}>
                    {['Дома', 'ЧПП', 'Услуги', 'Локација', 'АИ Чет'].map(link => (
                        <button key={link} onClick={() => navigate('/')} style={{
                            background: 'none', border: 'none', color: '#93c5fd',
                            fontSize: '0.9rem', cursor: 'pointer', padding: '8px 14px',
                            borderRadius: 8, transition: 'all 0.2s'
                        }}>{link}</button>
                    ))}
                </div>

                {/* Најава Button */}
                <button onClick={() => navigate('/login')} style={{
                    padding: '9px 24px', borderRadius: 8,
                    border: '2px solid #fff', background: 'transparent',
                    color: '#fff', fontWeight: 700, fontSize: '0.9rem', cursor: 'pointer'
                }}>
                    Најава
                </button>
            </div>

            {/* Hero Section */}
            <div style={{ background: '#1B3A6B', padding: '48px 16px', textAlign: 'center' }}>
                <div style={{
                    width: 56, height: 56, borderRadius: '50%',
                    background: 'rgba(212,160,23,0.2)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    margin: '0 auto 16px auto', fontSize: 28
                }}>🛡️</div>
                <h1 style={{ color: '#fff', fontSize: '2rem', fontWeight: 700, margin: '0 0 8px 0' }}>
                    Пристап до е-Влада
                </h1>
                <p style={{ color: '#93c5fd', margin: 0, fontSize: '0.95rem' }}>
                    Безбедна автентикација за граѓани и службеници
                </p>
            </div>

            <div style={{ maxWidth: 900, margin: '0 auto', padding: '40px 16px' }}>

                {/* Tabs */}
                <div style={{
                    display: 'flex', borderRadius: 12, overflow: 'hidden',
                    border: '1px solid #e2e8f0', background: '#fff',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.06)', marginBottom: 32
                }}>
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => switchTab(tab.id)}
                            style={{
                                flex: 1, padding: '14px 0', border: 'none', cursor: 'pointer',
                                fontSize: '0.9rem', transition: 'all 0.2s',
                                background: activeTab === tab.id ? '#1B3A6B' : 'transparent',
                                color: activeTab === tab.id ? '#fff' : '#64748b',
                                fontWeight: activeTab === tab.id ? 600 : 400,
                            }}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Login Tab */}
                {activeTab === 'login' && (
                    <div style={{ maxWidth: 440, margin: '0 auto' }}>
                        <div style={{
                            background: '#fff', borderRadius: 16, padding: 32,
                            border: '1px solid #e2e8f0', boxShadow: '0 1px 4px rgba(0,0,0,0.06)'
                        }}>
                            <h2 style={{ textAlign: 'center', color: '#1e293b', fontSize: '1.25rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                                Добредојдовте
                            </h2>
                            <p style={{ textAlign: 'center', color: '#64748b', fontSize: '0.875rem', margin: '0 0 28px 0' }}>
                                Најавете се со вашите акредитиви
                            </p>

                            {loginSuccess ? (
                                <div style={{ textAlign: 'center', padding: '24px 0' }}>
                                    <div style={{
                                        width: 64, height: 64, borderRadius: '50%',
                                        background: '#F0FDF4', display: 'flex',
                                        alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 16px auto', fontSize: 32
                                    }}>✅</div>
                                    <h3 style={{ color: '#1e293b', margin: '0 0 8px 0' }}>Успешна Најава!</h3>
                                    <p style={{ color: '#64748b', fontSize: '0.875rem', margin: 0 }}>
                                        Добредојдовте на е-Влада порталот.
                                    </p>
                                </div>
                            ) : (
                                <form onSubmit={handleLogin}>
                                    {error && (
                                        <div style={{
                                            background: '#FFF1F2', border: '1px solid #fecdd3',
                                            borderRadius: 8, padding: '10px 14px', marginBottom: 16,
                                            color: '#CE2028', fontSize: '0.85rem'
                                        }}>{error}</div>
                                    )}

                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                            Е-маил адреса
                                        </label>
                                        <div style={{ position: 'relative' }}>
                                            <span style={{
                                                position: 'absolute', left: 12, top: '50%',
                                                transform: 'translateY(-50%)', color: '#94a3b8', fontSize: 16
                                            }}>✉️</span>
                                            <input
                                                type="email"
                                                required
                                                value={loginForm.email}
                                                onChange={(e) => setLoginForm({ ...loginForm, email: e.target.value })}
                                                placeholder="ime@primer.mk"
                                                style={{
                                                    width: '100%', paddingLeft: 38, paddingRight: 14,
                                                    paddingTop: 12, paddingBottom: 12, borderRadius: 10,
                                                    border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                    fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                }}
                                            />
                                        </div>
                                    </div>

                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                            Лозинка
                                        </label>
                                        <div style={{ position: 'relative' }}>
                                            <span style={{
                                                position: 'absolute', left: 12, top: '50%',
                                                transform: 'translateY(-50%)', color: '#94a3b8', fontSize: 16
                                            }}>🔒</span>
                                            <input
                                                type={showPassword ? 'text' : 'password'}
                                                required
                                                value={loginForm.password}
                                                onChange={(e) => setLoginForm({ ...loginForm, password: e.target.value })}
                                                placeholder="Внесете лозинка"
                                                style={{
                                                    width: '100%', paddingLeft: 38, paddingRight: 44,
                                                    paddingTop: 12, paddingBottom: 12, borderRadius: 10,
                                                    border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                    fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                }}
                                            />
                                            <button
                                                type="button"
                                                onClick={() => setShowPassword(!showPassword)}
                                                style={{
                                                    position: 'absolute', right: 12, top: '50%',
                                                    transform: 'translateY(-50%)', background: 'none',
                                                    border: 'none', cursor: 'pointer', color: '#94a3b8', fontSize: 16
                                                }}
                                            >
                                                {showPassword ? '🙈' : '👁️'}
                                            </button>
                                        </div>
                                    </div>

                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.875rem', color: '#475569' }}>
                                            <input type="checkbox" style={{ borderRadius: 4 }} />
                                            Запомни ме
                                        </label>
                                        <button type="button" style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#1B3A6B', fontWeight: 600, fontSize: '0.875rem' }}>
                                            Заборавена лозинка?
                                        </button>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={loading}
                                        style={{
                                            width: '100%', padding: '12px 0', borderRadius: 10, border: 'none',
                                            background: '#1B3A6B', color: '#fff', fontWeight: 600,
                                            fontSize: '0.9rem', cursor: loading ? 'not-allowed' : 'pointer',
                                            opacity: loading ? 0.7 : 1, marginBottom: 16
                                        }}
                                    >
                                        {loading ? 'Се вчитува...' : 'Најава'}
                                    </button>

                                    <div style={{ display: 'flex', alignItems: 'center', margin: '8px 0 16px 0' }}>
                                        <div style={{ flex: 1, height: 1, background: '#e2e8f0' }} />
                                        <span style={{ padding: '0 12px', color: '#94a3b8', fontSize: '0.8rem' }}>или</span>
                                        <div style={{ flex: 1, height: 1, background: '#e2e8f0' }} />
                                    </div>

                                    <button
                                        type="button"
                                        onClick={() => switchTab('register')}
                                        style={{
                                            width: '100%', padding: '12px 0', borderRadius: 10,
                                            border: '1px solid #e2e8f0', background: 'transparent',
                                            color: '#475569', fontSize: '0.875rem', cursor: 'pointer'
                                        }}
                                    >
                                        Немате сметка? Регистрирајте се
                                    </button>
                                </form>
                            )}
                        </div>

                        {/* Security notice */}
                        <div style={{
                            marginTop: 16, padding: 16, borderRadius: 12,
                            background: '#FFFBEB', border: '1px solid #FDE68A',
                            display: 'flex', alignItems: 'flex-start', gap: 10
                        }}>
                            <span style={{ fontSize: 16, flexShrink: 0 }}>🛡️</span>
                            <p style={{ margin: 0, fontSize: '0.75rem', color: '#92400E', lineHeight: 1.6 }}>
                                е-Влада користи 256-битно SSL шифрирање. Никогаш не ги споделувајте вашите акредитиви со трети лица.
                            </p>
                        </div>
                    </div>
                )}

                {/* Register Tab */}
                {activeTab === 'register' && (
                    <div style={{ maxWidth: 520, margin: '0 auto' }}>
                        <div style={{
                            background: '#fff', borderRadius: 16, padding: 32,
                            border: '1px solid #e2e8f0', boxShadow: '0 1px 4px rgba(0,0,0,0.06)'
                        }}>
                            <h2 style={{ textAlign: 'center', color: '#1e293b', fontSize: '1.25rem', fontWeight: 700, margin: '0 0 4px 0' }}>
                                Создадете Сметка
                            </h2>
                            <p style={{ textAlign: 'center', color: '#64748b', fontSize: '0.875rem', margin: '0 0 28px 0' }}>
                                Регистрирајте се за пристап до сите услуги
                            </p>

                            {registerSuccess ? (
                                <div style={{ textAlign: 'center', padding: '24px 0' }}>
                                    <div style={{
                                        width: 64, height: 64, borderRadius: '50%',
                                        background: '#F0FDF4', display: 'flex',
                                        alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 16px auto', fontSize: 32
                                    }}>✅</div>
                                    <h3 style={{ color: '#1e293b', margin: '0 0 8px 0' }}>Регистрацијата е успешна!</h3>
                                    <p style={{ color: '#64748b', fontSize: '0.875rem', margin: '0 0 20px 0' }}>
                                        Проверете го вашиот е-маил за потврда на сметката.
                                    </p>
                                    <button
                                        onClick={() => switchTab('login')}
                                        style={{
                                            padding: '10px 24px', borderRadius: 8, border: 'none',
                                            background: '#1B3A6B', color: '#fff', fontWeight: 600,
                                            fontSize: '0.875rem', cursor: 'pointer'
                                        }}
                                    >
                                        Кон Најава
                                    </button>
                                </div>
                            ) : (
                                <form onSubmit={handleRegister}>
                                    {error && (
                                        <div style={{
                                            background: '#FFF1F2', border: '1px solid #fecdd3',
                                            borderRadius: 8, padding: '10px 14px', marginBottom: 16,
                                            color: '#CE2028', fontSize: '0.85rem'
                                        }}>{error}</div>
                                    )}

                                    {/* Name */}
                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                            Име и Презиме
                                        </label>
                                        <input
                                            type="text"
                                            required
                                            value={registerForm.name}
                                            onChange={(e) => setRegisterForm({ ...registerForm, name: e.target.value })}
                                            placeholder="Вашето полно ime"
                                            style={{
                                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                                border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                            }}
                                        />
                                    </div>

                                    {/* EMBG */}
                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                            ЕМБГ
                                        </label>
                                        <input
                                            type="text"
                                            required
                                            maxLength={13}
                                            value={registerForm.embg}
                                            onChange={(e) => setRegisterForm({ ...registerForm, embg: e.target.value })}
                                            placeholder="Единствен матичен број на граѓанинот"
                                            style={{
                                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                                border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                            }}
                                        />
                                    </div>

                                    {/* Email */}
                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                            Е-маил
                                        </label>
                                        <input
                                            type="email"
                                            required
                                            value={registerForm.email}
                                            onChange={(e) => setRegisterForm({ ...registerForm, email: e.target.value })}
                                            placeholder="ime@primer.mk"
                                            style={{
                                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                                border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                            }}
                                        />
                                    </div>

                                    {/* Password + Confirm */}
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                                        <div>
                                            <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                                Лозинка
                                            </label>
                                            <input
                                                type="password"
                                                required
                                                value={registerForm.password}
                                                onChange={(e) => setRegisterForm({ ...registerForm, password: e.target.value })}
                                                placeholder="Лозинка"
                                                style={{
                                                    width: '100%', padding: '12px 14px', borderRadius: 10,
                                                    border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                    fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                }}
                                            />
                                        </div>
                                        <div>
                                            <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                                Потврди
                                            </label>
                                            <input
                                                type="password"
                                                required
                                                value={registerForm.confirm}
                                                onChange={(e) => setRegisterForm({ ...registerForm, confirm: e.target.value })}
                                                placeholder="Потврди"
                                                style={{
                                                    width: '100%', padding: '12px 14px', borderRadius: 10,
                                                    border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                    fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                }}
                                            />
                                        </div>
                                    </div>

                                    {/* Role selector */}
                                    <div style={{ marginBottom: 16 }}>
                                        <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 8 }}>
                                            Тип на корисник
                                        </label>
                                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                                            {roles.map((role) => (
                                                <button
                                                    key={role.id}
                                                    type="button"
                                                    onClick={() => setSelectedRole(role.id)}
                                                    style={{
                                                        padding: '10px 6px', borderRadius: 10, cursor: 'pointer',
                                                        border: `1px solid ${selectedRole === role.id ? role.color : '#e2e8f0'}`,
                                                        background: selectedRole === role.id ? role.bg : 'transparent',
                                                        color: selectedRole === role.id ? role.color : '#64748b',
                                                        fontWeight: selectedRole === role.id ? 600 : 400,
                                                        fontSize: '0.8rem', textAlign: 'center', transition: 'all 0.2s'
                                                    }}
                                                >
                                                    <div style={{ fontSize: 18, marginBottom: 4 }}>{role.icon}</div>
                                                    {role.title}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Terms */}
                                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 8, cursor: 'pointer', marginBottom: 20 }}>
                                        <input type="checkbox" required style={{ marginTop: 2 }} />
                                        <span style={{ fontSize: '0.78rem', color: '#475569' }}>
                                            Се согласувам со{' '}
                                            <span style={{ textDecoration: 'underline', color: '#1B3A6B', cursor: 'pointer' }}>Условите за користење</span>
                                            {' '}и{' '}
                                            <span style={{ textDecoration: 'underline', color: '#1B3A6B', cursor: 'pointer' }}>Политиката за приватност</span>
                                        </span>
                                    </label>

                                    <button
                                        type="submit"
                                        disabled={loading}
                                        style={{
                                            width: '100%', padding: '12px 0', borderRadius: 10, border: 'none',
                                            background: '#1B3A6B', color: '#fff', fontWeight: 600,
                                            fontSize: '0.9rem', cursor: loading ? 'not-allowed' : 'pointer',
                                            opacity: loading ? 0.7 : 1
                                        }}
                                    >
                                        {loading ? 'Се вчитува...' : 'Регистрирај се'}
                                    </button>
                                </form>
                            )}
                        </div>
                    </div>
                )}

                {/* Roles Tab */}
                {activeTab === 'roles' && (
                    <div>
                        <div style={{ textAlign: 'center', marginBottom: 32 }}>
                            <h2 style={{ color: '#1e293b', fontSize: '1.25rem', fontWeight: 700, margin: '0 0 8px 0' }}>
                                Типови на Корисници
                            </h2>
                            <p style={{ color: '#64748b', fontSize: '0.875rem', margin: 0 }}>
                                Изберете го вашиот тип на пристап во системот
                            </p>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 24 }}>
                            {roles.map((role) => (
                                <div
                                    key={role.id}
                                    onClick={() => setSelectedRole(role.id)}
                                    style={{
                                        background: '#fff', borderRadius: 16, padding: 24, textAlign: 'center',
                                        border: `${selectedRole === role.id ? 2 : 1}px solid ${selectedRole === role.id ? role.color : '#e2e8f0'}`,
                                        boxShadow: '0 1px 4px rgba(0,0,0,0.06)', cursor: 'pointer',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    <div style={{
                                        width: 64, height: 64, borderRadius: 16,
                                        background: role.bg, display: 'flex',
                                        alignItems: 'center', justifyContent: 'center',
                                        margin: '0 auto 16px auto', fontSize: 32
                                    }}>
                                        {role.icon}
                                    </div>
                                    <h3 style={{ color: '#1e293b', fontSize: '1.1rem', fontWeight: 700, margin: '0 0 8px 0' }}>
                                        {role.title}
                                    </h3>
                                    <p style={{ color: '#64748b', fontSize: '0.875rem', lineHeight: 1.6, margin: '0 0 20px 0' }}>
                                        {role.desc}
                                    </p>

                                    <div style={{ textAlign: 'left', marginBottom: 20 }}>
                                        {role.features.map((f) => (
                                            <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                                                <span style={{ color: role.color, fontSize: 14, flexShrink: 0 }}>✓</span>
                                                <span style={{ fontSize: '0.8rem', color: '#374151' }}>{f}</span>
                                            </div>
                                        ))}
                                    </div>

                                    <button
                                        onClick={(e) => { e.stopPropagation(); setSelectedRole(role.id); switchTab('register'); }}
                                        style={{
                                            width: '100%', padding: '10px 0', borderRadius: 10, border: 'none',
                                            background: role.color, color: '#fff', fontWeight: 600,
                                            fontSize: '0.875rem', cursor: 'pointer'
                                        }}
                                    >
                                        Регистрирај се
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Login;