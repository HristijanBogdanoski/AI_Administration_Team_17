import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Profile() {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');
    const [embgLocked, setEmbgLocked] = useState(false);
    const [form, setForm] = useState({
        email: '',
        embg: '',
        currentPassword: '',
        newPassword: '',
        confirmNew: '',
    });

    useEffect(() => {
        const fetchMe = async () => {
            const token = localStorage.getItem('token');
            if (!token) {
                setLoading(false);
                return;
            }

            try {
                setLoading(true);
                const response = await fetch('http://127.0.0.1:8000/auth/me', {
                    headers: { Authorization: `Bearer ${token}` },
                });
                const data = await response.json();

                if (!response.ok) {
                    setError(data.detail || 'Неуспешно вчитување на профил.');
                    return;
                }

                setForm((prev) => ({
                    ...prev,
                    email: data.email || '',
                    embg: data.embg || '',
                }));
                setEmbgLocked(Boolean(data.embg));
            } catch {
                setError('Грешка при поврзување со серверот.');
            } finally {
                setLoading(false);
            }
        };

        fetchMe();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setMessage('');

        const token = localStorage.getItem('token');
        if (!token) {
            setError('Потребно е да сте најавени.');
            return;
        }

        if (form.newPassword && form.newPassword !== form.confirmNew) {
            setError('Новите лозинки не се совпаѓаат.');
            return;
        }

        const payload = {};
        if (form.email) payload.email = form.email;
        if (form.newPassword) {
            payload.current_password = form.currentPassword;
            payload.new_password = form.newPassword;
        }
        if (!embgLocked && form.embg) payload.embg = form.embg;

        if (Object.keys(payload).length === 0) {
            setError('Нема промени за зачувување.');
            return;
        }

        try {
            setSaving(true);
            const response = await fetch('http://127.0.0.1:8000/auth/me', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();

            if (!response.ok) {
                setError(data.detail || 'Неуспешно ажурирање на профил.');
                return;
            }

            setForm((prev) => ({
                ...prev,
                email: data.email || prev.email,
                embg: data.embg || prev.embg,
                currentPassword: '',
                newPassword: '',
                confirmNew: '',
            }));
            if (data.embg) setEmbgLocked(true);
            setMessage('Профилот е успешно ажуриран.');
        } catch {
            setError('Грешка при поврзување со серверот.');
        } finally {
            setSaving(false);
        }
    };

    const hasToken = Boolean(localStorage.getItem('token'));

    if (!hasToken) {
        return (
            <div style={{ maxWidth: 520, margin: '40px auto', padding: '0 16px' }}>
                <div style={{ background: '#fff', borderRadius: 16, padding: 32, border: '1px solid #e2e8f0' }}>
                    <h2 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Поставки на профил</h2>
                    <p style={{ margin: '0 0 20px 0', color: '#64748b' }}>За пристап до профил, најавете се.</p>
                    <button
                        onClick={() => navigate('/login')}
                        style={{
                            padding: '10px 20px', borderRadius: 8, border: 'none',
                            background: '#1B3A6B', color: '#fff', cursor: 'pointer', fontWeight: 600
                        }}
                    >
                        Кон Најава
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div style={{ maxWidth: 560, margin: '40px auto', padding: '0 16px 60px' }}>
            <div style={{ background: '#fff', borderRadius: 16, padding: 32, border: '1px solid #e2e8f0' }}>
                <h2 style={{ margin: '0 0 8px 0', color: '#1e293b' }}>Поставки на профил</h2>
                <p style={{ margin: '0 0 24px 0', color: '#64748b', fontSize: '0.9rem' }}>
                    Променете е-маил, лозинка и поставете ЕМБГ (само еднаш).
                </p>

                {error && (
                    <div style={{
                        background: '#FFF1F2', border: '1px solid #fecdd3', borderRadius: 8,
                        padding: '10px 14px', marginBottom: 16, color: '#CE2028', fontSize: '0.85rem'
                    }}>
                        {error}
                    </div>
                )}

                {message && (
                    <div style={{
                        background: '#F0FDF4', border: '1px solid #86efac', borderRadius: 8,
                        padding: '10px 14px', marginBottom: 16, color: '#166534', fontSize: '0.85rem'
                    }}>
                        {message}
                    </div>
                )}

                <form onSubmit={handleSubmit}>
                    <div style={{ marginBottom: 16 }}>
                        <label style={{ display: 'block', marginBottom: 6, fontSize: '0.875rem', color: '#374151' }}>Е-маил</label>
                        <input
                            type="email"
                            value={form.email}
                            onChange={(e) => setForm({ ...form, email: e.target.value })}
                            style={{
                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                border: '1px solid #e2e8f0', background: '#F8FAFC', boxSizing: 'border-box'
                            }}
                        />
                    </div>

                    <div style={{ marginBottom: 16 }}>
                        <label style={{ display: 'block', marginBottom: 6, fontSize: '0.875rem', color: '#374151' }}>
                            ЕМБГ {embgLocked ? '(заклучено)' : '(опционално)'}
                        </label>
                        <input
                            type="text"
                            maxLength={13}
                            disabled={embgLocked}
                            value={form.embg}
                            onChange={(e) => setForm({ ...form, embg: e.target.value })}
                            placeholder="Единствен матичен број на граѓанинот"
                            style={{
                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                border: '1px solid #e2e8f0', background: embgLocked ? '#F1F5F9' : '#F8FAFC', boxSizing: 'border-box'
                            }}
                        />
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                        <div>
                            <label style={{ display: 'block', marginBottom: 6, fontSize: '0.875rem', color: '#374151' }}>Тековна лозинка</label>
                            <input
                                type="password"
                                value={form.currentPassword}
                                onChange={(e) => setForm({ ...form, currentPassword: e.target.value })}
                                placeholder="Потребна при промена"
                                style={{
                                    width: '100%', padding: '12px 14px', borderRadius: 10,
                                    border: '1px solid #e2e8f0', background: '#F8FAFC', boxSizing: 'border-box'
                                }}
                            />
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: 6, fontSize: '0.875rem', color: '#374151' }}>Нова лозинка</label>
                            <input
                                type="password"
                                value={form.newPassword}
                                onChange={(e) => setForm({ ...form, newPassword: e.target.value })}
                                placeholder="Оставете празно"
                                style={{
                                    width: '100%', padding: '12px 14px', borderRadius: 10,
                                    border: '1px solid #e2e8f0', background: '#F8FAFC', boxSizing: 'border-box'
                                }}
                            />
                        </div>
                    </div>

                    <div style={{ marginBottom: 20 }}>
                        <label style={{ display: 'block', marginBottom: 6, fontSize: '0.875rem', color: '#374151' }}>Потврди нова лозинка</label>
                        <input
                            type="password"
                            value={form.confirmNew}
                            onChange={(e) => setForm({ ...form, confirmNew: e.target.value })}
                            style={{
                                width: '100%', padding: '12px 14px', borderRadius: 10,
                                border: '1px solid #e2e8f0', background: '#F8FAFC', boxSizing: 'border-box'
                            }}
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading || saving}
                        style={{
                            width: '100%', padding: '12px 0', borderRadius: 10, border: 'none',
                            background: '#1B3A6B', color: '#fff', fontWeight: 600,
                            opacity: loading || saving ? 0.7 : 1, cursor: loading || saving ? 'not-allowed' : 'pointer'
                        }}
                    >
                        {loading ? 'Се вчитува...' : saving ? 'Се зачувува...' : 'Зачувај промени'}
                    </button>
                </form>
            </div>
        </div>
    );
}

export default Profile;
