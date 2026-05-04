import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';

const LocationsPage = () => {
  const routerLocation = useLocation();
  const [locations, setLocations] = useState([]);
  const [selectedLoc, setSelectedLoc] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);

  // --- CRUD STATE ---
  const [showCrudModal, setShowCrudModal] = useState(false);
  const [modalMode, setModalMode] = useState('create');
  const [selectedLocationForCrud, setSelectedLocationForCrud] = useState(null);
  const [formData, setFormData] = useState({
    service_id: '',
    service_name: '',
    office_name: '',
    address: '',
    coordinates: { lat: null, lng: null },
    working_hours: '',
    contact_email: '',
    notes: ''
  });
  const [isLoggedIn, setIsLoggedIn] = useState(!!localStorage.getItem("token"));
  const [isGeocoding, setIsGeocoding] = useState(false);
  const [geocodingError, setGeocodingError] = useState('');
  const [availableServices, setAvailableServices] = useState([]);

  // --- FETCH SERVICES FOR DROPDOWN ---
  useEffect(() => {
    const fetchServices = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/services');
        if (response.ok) {
          const services = await response.json();
          console.log("Services loaded:", services[0]); // log first item to see field names
          setAvailableServices(services);
        }
      } catch (err) {
        console.error('Error fetching services:', err);
      }
    };
    fetchServices();
  }, []);

  // --- GEOCODING HANDLER ---
  const handleGeocode = async () => {
    if (!formData.office_name && !formData.address) {
      setGeocodingError('Внесете име на канцеларија или адреса за автоматско геокодирање');
      return;
    }
    setIsGeocoding(true);
    setGeocodingError('');
    try {
      const institution = formData.office_name || formData.service_name;
      const address = formData.address;
      const response = await fetch(`http://127.0.0.1:8000/location/map?institution=${encodeURIComponent(institution)}&address=${encodeURIComponent(address)}`);
      if (response.ok) {
        const result = await response.json();
        setFormData(prev => ({
          ...prev,
          coordinates: {
            lat: parseFloat(result.coordinates.lat),
            lng: parseFloat(result.coordinates.lng)
          }
        }));
      } else {
        const error = await response.json();
        setGeocodingError(error.detail || 'Грешка при геокодирање');
      }
    } catch (err) {
      setGeocodingError('Грешка при конекција со геокодинг сервис');
    } finally {
      setIsGeocoding(false);
    }
  };

  // --- CRUD HANDLERS ---
  const handleCrud = async (mode, location = null) => {
    setModalMode(mode);
    setSelectedLocationForCrud(location);
    setGeocodingError('');
    if (mode === 'create') {
      setFormData({ service_id: '', service_name: '', office_name: '', address: '', coordinates: { lat: null, lng: null }, working_hours: '', contact_email: '', notes: '' });
    } else if (mode === 'edit' && location) {
      setFormData({
        service_id: location.service_id || '',
        service_name: location.service_name || '',
        office_name: location.office_name || '',
        address: location.address || '',
        working_hours: location.working_hours || '',
        coordinates: location.coordinates || { lat: null, lng: null },
        contact_email: location.contact_email || '',
        notes: location.notes || ''
      });
    }
    setShowCrudModal(true);
  };

  const submitCrud = async () => {
    let latVal = parseFloat(formData.coordinates?.lat);
    let lngVal = parseFloat(formData.coordinates?.lng);

    console.log("Form data:", formData);
  console.log("lat:", latVal, "lng:", lngVal);

    if (modalMode !== 'delete') {
      if (isNaN(latVal) || isNaN(lngVal)) {
        setGeocodingError('Invalid coordinates');
        return;
      }
      if (!formData.service_id?.trim() || !formData.service_name?.trim() || !formData.office_name?.trim() || !formData.address?.trim() || !formData.working_hours?.trim() || !formData.contact_email?.trim()) {
        setGeocodingError('Сите задолжителни полиња мораат да бидат пополнети');
        return;
      }
    }

    const token = localStorage.getItem("token");
    let url = "http://127.0.0.1:8000/location";
    let method = "POST";
    if (modalMode === 'edit') { url += `/${selectedLocationForCrud.id}`; method = "PUT"; }
    else if (modalMode === 'delete') { url += `/${selectedLocationForCrud.id}`; method = "DELETE"; }

    let dataToSend;
    if (modalMode !== 'delete') {
      dataToSend = {
        service_id: formData.service_id,
        service_name: formData.service_name,
        office_name: formData.office_name,
        address: formData.address,
        coordinates: {
          lat: latVal,
          lng: lngVal
        },
        working_hours: formData.working_hours,
        contact_email: formData.contact_email,
        notes: formData.notes || ''
      };
    }

    try {
      const response = await fetch(url, {
        method,
        headers: {"Content-Type": "application/json", "Accept": "application/json", "Authorization": `Bearer ${token}`},
        body: modalMode !== 'delete' ? JSON.stringify(dataToSend) : undefined
      });
      if (response.ok) {
        if (modalMode === 'delete') {
          setLocations(prev => prev.filter(l => l.id !== selectedLocationForCrud.id));
          if (selectedLoc?.id === selectedLocationForCrud.id) setSelectedLoc(null);
        } else {
          const result = await response.json();
          if (modalMode === 'create') setLocations(prev => [result, ...prev]);
          else {
            setLocations(prev => prev.map(l => l.id === selectedLocationForCrud.id ? result : l));
            if (selectedLoc?.id === selectedLocationForCrud.id) setSelectedLoc(result);
          }
        }
        setShowCrudModal(false);
      } else {
        const error = await response.json();
        setGeocodingError(error.detail || 'Грешка при зачувување');
      }
    } catch (err) { setGeocodingError('Грешка при конекција со серверот'); }
  };

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        setLoading(true);
        const res = await axios.get('http://127.0.0.1:8000/location');
        const data = res.data || [];
        // Sort by creation date or ID (newest first) to maintain order
        const sortedData = data.sort((a, b) => {
          // Try to sort by ID (assuming newer items have higher IDs)
          if (a.id && b.id) return b.id - a.id;
          // Fallback: keep original order if no sortable field
          return 0;
        });
        setLocations(sortedData);
        if (data.length > 0) {
          const requestedServiceId = routerLocation.state?.serviceId;
          const match = data.find((loc) => requestedServiceId && loc.service_id === requestedServiceId);
          setSelectedLoc(match || data[0]);
        }
      } catch (err) { console.error(err); } finally { setLoading(false); }
    };
    fetchLocations();
  }, [routerLocation.state]);

  const filteredLocations = locations.filter(loc =>
    loc.office_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    loc.service_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    loc.address.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getMapUrl = (loc) => {
    if (loc?.coordinates?.lat == null || loc?.coordinates?.lng == null) return "about:blank";
    return `https://www.google.com/maps?q=${loc.coordinates.lat},${loc.coordinates.lng}&z=15&output=embed`;
  };

  if (loading) return <div className="text-center mt-5 p-5 fw-bold">Се вчитуваат податоците од базата...</div>;

  return (
    <div style={{ backgroundColor: '#f4f7f9', minHeight: '100vh', fontFamily: "'Segoe UI', sans-serif" }}>
      <style>
        {`
          .header-section { background: #1a3a5f; color: white; padding: 60px 0 100px; text-align: center; }
          .search-container {
            max-width: 700px; margin: -35px auto 45px; background: white;
            padding: 12px 25px; border-radius: 50px; box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            display: flex; align-items: center; border: 1px solid #eee;
            position: relative; z-index: 100;
          }
          .search-input { border: none; outline: none; width: 100%; margin-left: 12px; font-size: 16px; }
          .list-card { background: white; border-radius: 15px; border: 1px solid #eee; margin-bottom: 12px; cursor: pointer; transition: 0.3s ease; }
          .list-card:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
          .list-card.active { border-left: 8px solid #1a3a5f; background: #f0f4f8; }
          .office-name-text { color: #1a3a5f; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
          .sticky-panel { position: -webkit-sticky; position: sticky; top: 25px; height: fit-content; }
          .map-box { height: 350px; border-radius: 20px 20px 0 0; overflow: hidden; background: #e5e3df; }
        `}
      </style>

      <div className="header-section">
        <div className="container">
          <h2 className="fw-bold display-6">Институции во Македонија</h2>
          <p className="opacity-75">Пронајдете ги локациите на државните служби</p>
        </div>
      </div>

      <div className="container">
        <div className="search-container">
          <i className="bi bi-search text-muted"></i>
          <input type="text" className="search-input" placeholder="Пребарајте услуга, град или име..." onChange={(e) => setSearchTerm(e.target.value)} />
        </div>

        {isLoggedIn && (
          <div style={{display: "flex", justifyContent: "center", marginTop: "15px"}}>
            <button onClick={() => handleCrud('create')} style={{ backgroundColor: "#1a3a5f", color: "#fff", border: "none", padding: "10px 20px", borderRadius: "20px", cursor: "pointer", fontWeight: "600", position:"relative", top:"-20px" }}>
              + Додади Локација
            </button>
          </div>
        )}

        <div className="row g-4 align-items-start">
          <div className="col-md-5">
            {filteredLocations.map(loc => (
              <div key={loc.id} className={`list-card p-4 shadow-sm ${selectedLoc?.id === loc.id ? 'active' : ''}`} onClick={() => setSelectedLoc(loc)} >
                <div style={{display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px"}}>
                  <div className="office-name-text">{loc.office_name}</div>
                  {isLoggedIn && (
                    <div style={{display: "flex", gap: "3px"}}>
                      <button onClick={(e) => {e.stopPropagation(); handleCrud('edit', loc);}} style={{ backgroundColor: "#3b82f6", color: "#fff", border: "none", padding: "2px 6px", borderRadius: "3px", cursor: "pointer", fontSize: "0.7rem" }}>Измени</button>
                      <button onClick={(e) => {e.stopPropagation(); handleCrud('delete', loc);}} style={{ backgroundColor: "#ef4444", color: "#fff", border: "none", padding: "2px 6px", borderRadius: "3px", cursor: "pointer", fontSize: "0.7rem" }}>Избриши</button>
                    </div>
                  )}
                </div>
                <div className="small fw-bold text-primary mb-2">{loc.service_name}</div>
                <div className="small text-muted"><i className="bi bi-geo-alt-fill me-1"></i> {loc.address}</div>
              </div>
            ))}
          </div>

          <div className="col-md-7 sticky-panel">
            {selectedLoc && (
              <div className="card border-0 shadow-lg" style={{ borderRadius: '20px' }}>
                <div className="map-box">
                  <iframe key={selectedLoc.id} title="map" width="100%" height="100%" frameBorder="0" src={getMapUrl(selectedLoc)} ></iframe>
                </div>
                <div className="card-body p-4">
                  <h4 className="fw-bold mb-1" style={{ color: '#1a3a5f' }}>{selectedLoc.office_name}</h4>
                  <span className="badge bg-light text-primary border mb-4 px-3 py-2 rounded-pill">{selectedLoc.service_name}</span>
                  <div className="row g-3">
                    <div className="col-6"><div className="p-3 bg-light rounded-4 h-100"><small className="text-muted fw-bold d-block mb-1">ЛОКАЦИЈА</small><span className="small fw-medium">{selectedLoc.address}</span></div></div>
                    <div className="col-6"><div className="p-3 bg-light rounded-4 h-100"><small className="text-muted fw-bold d-block mb-1">РАБОТНО ВРЕМЕ</small><span className="small fw-medium">{selectedLoc.working_hours}</span></div></div>
                  </div>
                  <div className="mt-4 d-flex gap-3">
                    <button className="btn py-3 flex-grow-1 text-white fw-bold shadow-sm" style={{ background: '#1a3a5f', borderRadius: '15px' }}>Јавете се</button>
                    <button className="btn btn-outline-dark py-3 px-4 shadow-sm" style={{ borderRadius: '15px' }}>Насоки</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* --- CRUD MODAL --- */}
      {showCrudModal && (
        <div style={{position: "fixed", top: 0, left: 0, right: 0, bottom: 0, backgroundColor: "rgba(0,0,0,0.6)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 1000}} onClick={() => setShowCrudModal(false)}>
          <div style={{backgroundColor: "#fff", padding: "30px", borderRadius: "12px", width: "100%", maxWidth: modalMode === 'delete' ? "400px" : "500px", position: "relative"}} onClick={(e) => e.stopPropagation()}>
            <button style={{position: "absolute", top: "10px", right: "10px", border: "none", background: "none", fontSize: "20px", cursor: "pointer"}} onClick={() => setShowCrudModal(false)}>&times;</button>
            <h3 style={{marginBottom: "20px", color: modalMode === 'delete' ? "#dc2626" : "#1e293b"}}>{modalMode === 'create' ? 'Нова Локација' : modalMode === 'edit' ? 'Ажурирај Локација' : 'Избриши Локација'}</h3>

            {modalMode === 'delete' ? (
              <div>
                <p style={{marginBottom: "20px", color: "#6b7280"}}>Дали сте сигурни дека сакате да ја избришете локацијата "<strong>{selectedLocationForCrud?.office_name}</strong>"?</p>
                <div style={{display: "flex", gap: "10px"}}>
                  <button onClick={submitCrud} style={{flex: 1, padding: "10px", backgroundColor: "#dc2626", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Избриши</button>
                  <button onClick={() => setShowCrudModal(false)} style={{flex: 1, padding: "10px", backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Откажи</button>
                </div>
              </div>
            ) : (
              <form onSubmit={(e) => {e.preventDefault(); submitCrud();}} style={{display: "flex", flexDirection: "column", gap: "15px"}}>
                <select 
                  value={formData.service_id} 
                  onChange={(e) => {
                    const selected = availableServices.find(s => s.service_id === e.target.value);
                    console.log("Selected service:", selected); // temporary debug
                    setFormData({ 
                      ...formData, 
                      service_id: e.target.value, 
                      service_name: selected?.name || '',
                      // Auto-fill location from service if available
                      address: selected?.location || formData.address,
                      // Auto-generate office name based on service name if empty
                      office_name: formData.office_name || (selected?.name ? `Главна канцеларија - ${selected.name}` : '')
                    });
                  }}
                  style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px", width: "100%"}} 
                  required
                >
                  <option value="">Изберете услуга...</option>
                  {availableServices.map(service => (
                    <option key={service.service_id} value={service.service_id}>
                      {service.service_name} ({service.service_id})
                    </option>
                  ))}
                </select>
                                <input type="text" placeholder="Име на Канцеларија *" value={formData.office_name} onChange={(e) => setFormData({...formData, office_name: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} required />
                <input type="text" placeholder="Адреса *" value={formData.address} onChange={(e) => setFormData({...formData, address: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} required />
                
                <button 
                  type="button" 
                  onClick={handleGeocode} 
                  disabled={isGeocoding}
                  style={{
                    padding: "10px",
                    backgroundColor: isGeocoding ? '#9ca3af' : '#059669',
                    color: "#fff",
                    border: "none",
                    borderRadius: "6px",
                    cursor: isGeocoding ? 'not-allowed' : 'pointer',
                    fontWeight: "500",
                    fontSize: "0.9rem"
                  }}
                >
                  {isGeocoding ? 'Геокодирање...' : '📍 Автоматски земи координати'}
                </button>
                
                <input type="email" placeholder="Контакт Е-пошта *" value={formData.contact_email} onChange={(e) => setFormData({...formData, contact_email: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} required />
                <input type="text" placeholder="Работно време *" value={formData.working_hours} onChange={(e) => setFormData({...formData, working_hours: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} required />
                <div style={{display: "flex", gap: "10px", marginBottom: "10px"}}>
                  <input 
                    type="number" 
                    step="any" 
                    placeholder="Латитуда *" 
                    value={formData.coordinates?.lat || ''} 
                    onChange={(e) => setFormData({...formData, coordinates: {...formData.coordinates, lat: parseFloat(e.target.value) || null}})} 
                    readOnly={formData.coordinates?.lat !== null && formData.coordinates?.lng !== null}
                    style={{
                      flex: 1, 
                      padding: "10px", 
                      border: formData.coordinates?.lat && formData.coordinates?.lng ? "2px solid #10b981" : "1px solid #d1d5db", 
                      borderRadius: "6px",
                      backgroundColor: formData.coordinates?.lat && formData.coordinates?.lng ? "#f0fdf4" : "#fff",
                      color: formData.coordinates?.lat && formData.coordinates?.lng ? "#065f46" : "inherit"
                    }} 
                    required 
                  />
                  <input 
                    type="number" 
                    step="any" 
                    placeholder="Лонгитуда *" 
                    value={formData.coordinates?.lng || ''} 
                    onChange={(e) => setFormData({...formData, coordinates: {...formData.coordinates, lng: parseFloat(e.target.value) || null}})} 
                    readOnly={formData.coordinates?.lat !== null && formData.coordinates?.lng !== null}
                    style={{
                      flex: 1, 
                      padding: "10px", 
                      border: formData.coordinates?.lat && formData.coordinates?.lng ? "2px solid #10b981" : "1px solid #d1d5db", 
                      borderRadius: "6px",
                      backgroundColor: formData.coordinates?.lat && formData.coordinates?.lng ? "#f0fdf4" : "#fff",
                      color: formData.coordinates?.lat && formData.coordinates?.lng ? "#065f46" : "inherit"
                    }} 
                    required 
                  />
                </div>
                {geocodingError && (
                  <div style={{
                    padding: "8px",
                    backgroundColor: '#fef2f2',
                    border: '1px solid #fecaca',
                    borderRadius: "6px",
                    color: '#dc2626',
                    fontSize: '0.8rem'
                  }}>
                    {geocodingError}
                  </div>
                )}
                
                {formData.coordinates?.lat && formData.coordinates?.lng && (
                  <div style={{
                    padding: "8px",
                    backgroundColor: '#f0fdf4',
                    border: '1px solid #bbf7d0',
                    borderRadius: "6px",
                    color: '#166534',
                    fontSize: '0.8rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    ✅ Координатите се автоматски земени: {formData.coordinates.lat.toFixed(6)}, {formData.coordinates.lng.toFixed(6)}
                  </div>
                )}
                <textarea placeholder="Белешки" value={formData.notes} onChange={(e) => setFormData({...formData, notes: e.target.value})} style={{padding: "10px", border: "1px solid #d1d5db", borderRadius: "6px"}} />
                <div style={{display: "flex", gap: "10px"}}>
                  <button type="submit" style={{flex: 1, padding: "10px", backgroundColor: "#1a3a5f", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>{modalMode === 'edit' ? 'Ажурирај' : 'Креирај'}</button>
                  <button type="button" onClick={() => setShowCrudModal(false)} style={{flex: 1, padding: "10px", backgroundColor: "#6b7280", color: "#fff", border: "none", borderRadius: "6px", cursor: "pointer"}}>Откажи</button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default LocationsPage;