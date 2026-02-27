import { useEffect, useState } from 'react';
import { Routes, Route, Link, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Network from './pages/Network';
import Map from './pages/Map';
import MapV2 from './pages/MapV2';
import ExpertFinder from './pages/ExpertFinder';
import Profile from './pages/Profile';
import Report from './pages/Report';
import { BarChart3, Users, Map as MapIcon, SearchCode, BookText, Menu, X } from 'lucide-react';
import './App.css';

function App() {
  const location = useLocation();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth > 1024) {
        setMobileNavOpen(false);
      }
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const navItems = [
    { path: '/', label: 'Dashboard', icon: BarChart3 },
    { path: '/report', label: 'Report', icon: BookText },
    { path: '/map', label: 'Map', icon: MapIcon },
    { path: '/network', label: 'Neighbours', icon: Users },
    { path: '/expert-search', label: 'Expert Finder', icon: SearchCode },
  ];

  return (
    <div className={`app-layout ${mobileNavOpen ? 'mobile-nav-open' : ''}`}>
      <button
        type="button"
        className={`sidebar-backdrop ${mobileNavOpen ? 'show' : ''}`}
        aria-label="Close navigation menu"
        onClick={() => setMobileNavOpen(false)}
      />

      <header className="mobile-topbar">
        <button
          type="button"
          className="sidebar-toggle"
          onClick={() => setMobileNavOpen((value) => !value)}
          aria-expanded={mobileNavOpen}
          aria-label={mobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
        >
          {mobileNavOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        <div className="mobile-topbar-brand">
          <h1 className="brand-title">TRISS</h1>
          <p className="brand-subtitle">Knowledge Portal</p>
        </div>
      </header>

      {/* Sidebar Navigation */}
      <nav className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-brand-wrap">
            <h1 className="brand-title">TRISS</h1>
            <p className="brand-subtitle">Knowledge Portal</p>
          </div>
          <button
            type="button"
            className="sidebar-close"
            onClick={() => setMobileNavOpen(false)}
            aria-label="Close navigation menu"
          >
            <X size={18} />
          </button>
        </div>

        <div className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${isActive ? 'active' : ''}`}
                onClick={() => setMobileNavOpen(false)}
              >
                <Icon size={20} className="nav-icon" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>

        <div className="sidebar-footer">
          <p>Powered by TRISS Architecture</p>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/network" element={<Network />} />
          <Route path="/map" element={<Map />} />
          <Route path="/map-v2" element={<MapV2 />} />
          <Route path="/expert-search" element={<ExpertFinder />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/report" element={<Report />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
