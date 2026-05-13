    import React, { useState } from 'react';
    import { useNavigate } from 'react-router-dom';

    function Login() {
        const navigate = useNavigate();
        const [activeTab, setActiveTab] = useState('login');
        const [showPassword, setShowPassword] = useState(false);
        const [loginForm, setLoginForm] = useState({ email: '', password: '' });
        const [registerForm, setRegisterForm] = useState({ name: '', email: '', password: '', confirm: '', embg: '', address: '', phone_number: '', gender: '' });
        const [loginSuccess, setLoginSuccess] = useState(false);
        const [registerSuccess, setRegisterSuccess] = useState(false);
        const [error, setError] = useState('');
        const [loading, setLoading] = useState(false);

        const loginAfterRegister = async (email, password) => {
            const loginResponse = await fetch('http://127.0.0.1:8000/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const loginData = await loginResponse.json();

            if (!loginResponse.ok) {
                throw new Error(loginData.detail || 'Регистрацијата е успешна, но најава не успеа.');
            }

            localStorage.setItem('token', loginData.access_token);
        };

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
                    body: JSON.stringify({ email: loginForm.email, password: loginForm.password }),
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
            if (registerForm.password.length < 8) {
                setError('Лозинката мора да содржи најмалку 8 карактери.');
                return;
            }
            setLoading(true);
            try {
                const response = await fetch('http://127.0.0.1:8000/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: registerForm.email,
                        full_name: registerForm.name,
                        embg: registerForm.embg || null,
                        address: registerForm.address || null,
                        phone_number: registerForm.phone_number || null,
                        gender: registerForm.gender || null,
                        password: registerForm.password,
                    }),
                });
                const data = await response.json();
                if (response.ok) {
                    await loginAfterRegister(registerForm.email, registerForm.password);
                    setRegisterSuccess(true);
                    setTimeout(() => navigate('/'), 1500);
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
        ];

        return (
            <div style={{ minHeight: '100vh', background: '#F4F6FA' }}>

                {/* Hero Section */}
                <div style={{ background: '#1B3A6B', padding: '48px 16px', textAlign: 'center' }}>
                    <div style={{
                        width: 56, height: 56, borderRadius: '50%',
                        background: 'rgba(212,160,23,0.2)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        margin: '0 auto 16px auto', fontSize: 28
                    }}>🛡️</div>
                    <h1 style={{ color: '#fff', fontSize: '2rem', fontWeight: 700, margin: '0 0 8px 0' }}>
                        Пристап до БрзиУслуги
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
                                            Добредојдовте на БрзиУслуги порталот.
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
                                    БрзиУслуги користи 256-битно SSL шифрирање. Никогаш не ги споделувајте вашите акредитиви со трети лица.
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
                                                ЕМБГ (опционално)
                                            </label>
                                            <input
                                                type="text"
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

                                        <div style={{ marginBottom: 16 }}>
                                            <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                                Адреса (опционално)
                                            </label>
                                            <input
                                                type="text"
                                                value={registerForm.address}
                                                onChange={(e) => setRegisterForm({ ...registerForm, address: e.target.value })}
                                                placeholder="Улица, број, град"
                                                style={{
                                                    width: '100%', padding: '12px 14px', borderRadius: 10,
                                                    border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                    fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                }}
                                            />
                                        </div>

                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                                            <div>
                                                <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                                    Телефон (опционално)
                                                </label>
                                                <input
                                                    type="text"
                                                    value={registerForm.phone_number}
                                                    onChange={(e) => setRegisterForm({ ...registerForm, phone_number: e.target.value })}
                                                    placeholder="+389 ..."
                                                    style={{
                                                        width: '100%', padding: '12px 14px', borderRadius: 10,
                                                        border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                        fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box'
                                                    }}
                                                />
                                            </div>
                                            <div>
                                                <label style={{ display: 'block', fontSize: '0.875rem', color: '#374151', marginBottom: 6 }}>
                                                    Пол (опционално)
                                                </label>
                                                <select
                                                    value={registerForm.gender}
                                                    onChange={(e) => setRegisterForm({ ...registerForm, gender: e.target.value })}
                                                    style={{
                                                        width: '100%', padding: '12px 14px', borderRadius: 10,
                                                        border: '1px solid #e2e8f0', background: '#F8FAFC',
                                                        fontSize: '0.875rem', outline: 'none', boxSizing: 'border-box',
                                                        cursor: 'pointer'
                                                    }}
                                                >
                                                    <option value="">—</option>
                                                    <option value="Машки">Машки</option>
                                                    <option value="Женски">Женски</option>
                                                </select>
                                            </div>
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

                </div>
            </div>
        );
    }

    export default Login;