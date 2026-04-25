import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useLocation } from 'react-router-dom';

const LocationsPage = () => {
  const routerLocation = useLocation();
  const [locations, setLocations] = useState([]);
  const [selectedLoc, setSelectedLoc] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLocations = async () => {
      try {
        setLoading(true);
        // Влечење податоци од твојот FastAPI бекенд
        const res = await axios.get('http://127.0.0.1:8000/location');
        const data = res.data || [];
        setLocations(data);

        if (data.length > 0) {
          const requestedServiceId = routerLocation.state?.serviceId;
          const requestedServiceName = routerLocation.state?.serviceName;

          const match = data.find((loc) => {
            const byId = requestedServiceId && loc.service_id === requestedServiceId;
            const byName = requestedServiceName && loc.service_name?.toLowerCase() === requestedServiceName.toLowerCase();
            return byId || byName;
          });

          setSelectedLoc(match || data[0]);
        }
      } catch (err) {
        console.error("Грешка при вчитување:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchLocations();
  }, [routerLocation.state]);

  // Филтрирање на листата
  const filteredLocations = locations.filter(loc => 
    loc.office_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    loc.service_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    loc.address.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // ГЕНЕРИРАЊЕ URL ЗА МАПА СО КООРДИНАТИ (lat,lng)
  const getMapUrl = (loc) => {
    const lat = loc?.coordinates?.lat;
    const lng = loc?.coordinates?.lng;

    if (lat == null || lng == null) {
      return "about:blank";
    }

    return `https://www.google.com/maps?q=${lat},${lng}&z=15&output=embed`;
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

          /* Лева листа - природна должина */
          .list-card { 
            background: white; border-radius: 15px; border: 1px solid #eee; 
            margin-bottom: 12px; cursor: pointer; transition: 0.3s ease;
          }
          .list-card:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
          .list-card.active { border-left: 8px solid #1a3a5f; background: #f0f4f8; }
          
          .office-name-text { color: #1a3a5f; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }

          /* ДЕСНА СТРАНА - СТРОГО STICKY */
          .sticky-panel {
            position: -webkit-sticky;
            position: sticky;
            top: 25px; 
            height: fit-content;
          }

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
        {/* Пребарувач */}
        <div className="search-container">
          <i className="bi bi-search text-muted"></i>
          <input 
            type="text" className="search-input" 
            placeholder="Пребарајте услуга, град или име..."
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        <div className="row g-4 align-items-start">
          
          {}
          <div className="col-md-5">
            {filteredLocations.map(loc => (
              <div 
                key={loc.id} 
                className={`list-card p-4 shadow-sm ${selectedLoc?.id === loc.id ? 'active' : ''}`}
                onClick={() => setSelectedLoc(loc)}
              >
                <div className="office-name-text">{loc.office_name}</div>
                <div className="small fw-bold text-primary mb-2">{loc.service_name}</div>
                <div className="small text-muted">
                   <i className="bi bi-geo-alt-fill me-1"></i> {loc.address}
                </div>
              </div>
            ))}
          </div>

          {}
          <div className="col-md-7 sticky-panel">
            {selectedLoc && (
              <div className="card border-0 shadow-lg" style={{ borderRadius: '20px' }}>
                <div className="map-box">
                  <iframe
                    key={selectedLoc.id} // Клучно за мапата да се освежи при промена
                    title="map" width="100%" height="100%" frameBorder="0"
                    src={getMapUrl(selectedLoc)}
                  ></iframe>
                </div>
                <div className="card-body p-4">
                  <h4 className="fw-bold mb-1" style={{ color: '#1a3a5f' }}>{selectedLoc.office_name}</h4>
                  <span className="badge bg-light text-primary border mb-4 px-3 py-2 rounded-pill">
                    {selectedLoc.service_name}
                  </span>
                  
                  <div className="row g-3">
                    <div className="col-6">
                      <div className="p-3 bg-light rounded-4 h-100">
                        <small className="text-muted fw-bold d-block mb-1">ЛОКАЦИЈА</small>
                        <span className="small fw-medium">{selectedLoc.address}</span>
                      </div>
                    </div>
                    <div className="col-6">
                      <div className="p-3 bg-light rounded-4 h-100">
                        <small className="text-muted fw-bold d-block mb-1">РАБОТНО ВРЕМЕ</small>
                        <span className="small fw-medium">{selectedLoc.working_hours}</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 d-flex gap-3">
                    <button className="btn py-3 flex-grow-1 text-white fw-bold shadow-sm" style={{ background: '#1a3a5f', borderRadius: '15px' }}>
                      Јавете се
                    </button>
                    <button className="btn btn-outline-dark py-3 px-4 shadow-sm" style={{ borderRadius: '15px' }}>
                      Насоки
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
          
        </div>
      </div>
    </div>
  );
};

export default LocationsPage;