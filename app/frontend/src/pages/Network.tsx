import React, { useState, useEffect } from 'react';
import {
    Search, Mail, X, UserCircle, Info
} from 'lucide-react';
import {
    searchResearchers, fetchResearcherProfile, fetchNeighbours,
    fetchMyPublications
} from '../api';
import type {
    ResearcherProfile, Neighbour, PublicationMatch, OwnPublication
} from '../api';
import '../App.css';
import { useEscapeClose } from '../hooks/useEscapeClose';

const Network: React.FC = () => {
    const [query, setQuery] = useState('');
    const [searchResults, setSearchResults] = useState<any[]>([]);
    const [selectedId, setSelectedId] = useState<number | null>(null);
    const [profile, setProfile] = useState<ResearcherProfile | null>(null);
    const [neighbours, setNeighbours] = useState<Neighbour[]>([]);
    const [loading, setLoading] = useState(false);

    // Filters
    const [kValue, setKValue] = useState(10);
    const [excludeDept, setExcludeDept] = useState(false);
    const [excludeSchool, setExcludeSchool] = useState(false);

    // Modals
    const [showMethodology, setShowMethodology] = useState(false);
    const [modalProfile, setModalProfile] = useState<ResearcherProfile | null>(null);
    const [modalPubs, setModalPubs] = useState<(PublicationMatch | OwnPublication)[]>([]);
    const [isOwnProfile, setIsOwnProfile] = useState(false);

    useEscapeClose(showMethodology, () => setShowMethodology(false));
    useEscapeClose(Boolean(modalProfile), () => setModalProfile(null));

    // Handle predictive search
    useEffect(() => {
        if (query.length < 2) {
            setSearchResults([]);
            return;
        }
        const timer = setTimeout(() => {
            searchResearchers(query).then(setSearchResults).catch(console.error);
        }, 200);
        return () => clearTimeout(timer);
    }, [query]);

    // Handle neighbor fetch when selection or filters change
    useEffect(() => {
        if (!selectedId) return;
        setLoading(true);
        // Fetch profile and neighbours
        Promise.all([
            fetchResearcherProfile(selectedId),
            fetchNeighbours(selectedId, kValue, excludeDept, excludeSchool)
        ])
            .then(([p, n]) => {
                setProfile(p);
                setNeighbours(n);
            })
            .catch(console.error)
            .finally(() => setLoading(false));
    }, [selectedId, kValue, excludeDept, excludeSchool]);

    const handleOpenProfile = async (targetId: number, isOwn: boolean = false) => {
        if (!selectedId && !isOwn) return;

        try {
            const p = await fetchResearcherProfile(targetId);
            // Always show the target researcher's own publications — consistent and complete
            const pubs = await fetchMyPublications(targetId);

            setModalProfile(p);
            setModalPubs(Array.isArray(pubs) ? pubs : []);
            setIsOwnProfile(isOwn);
        } catch (err) {
            console.error("Failed to open profile:", err);
        }
    };

    return (
        <div className="page-container fade-in">
            <header className="page-header">
                <div className="header-content">
                    <h1 className="page-title">Academic Neighbour</h1>
                    <p className="page-subtitle">Find your research neighbours based on semantic interest.</p>
                </div>
                <button className="method-btn" onClick={() => setShowMethodology(true)}>
                    <Info size={16} />
                    Methodology
                </button>
            </header>

            <div className="network-layout">
                {/* Search & Filter Column */}
                <div className="sidebar-col">
                    <div className="card search-card">
                        <h3 className="card-title">Search</h3>
                        <div className="search-box">
                            <Search className="search-icon" size={18} />
                            <input
                                type="text"
                                placeholder="Name or email..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className="search-input"
                            />
                        </div>
                        {searchResults.length > 0 && (
                            <div className="search-dropdown">
                                {searchResults.map(r => (
                                    <div key={r.n_id} className="dropdown-item" onClick={() => { setSelectedId(r.n_id); setQuery(''); setSearchResults([]); }}>
                                        <div className="item-name">{r.name}</div>
                                        <div className="item-meta">{r.department}</div>
                                    </div>
                                ))}
                            </div>
                        )}

                        <div className="filter-section">
                            <h3 className="card-title mt-4">Filters</h3>
                            <div className="filter-row">
                                <label>Neighbours:</label>
                                <select value={kValue} onChange={(e) => setKValue(Number(e.target.value))}>
                                    <option value={5}>5</option>
                                    <option value={10}>10</option>
                                    <option value={20}>20</option>
                                </select>
                            </div>
                            <div className="filter-checkbox">
                                <input type="checkbox" id="ex-dept" checked={excludeDept} onChange={e => setExcludeDept(e.target.checked)} />
                                <label htmlFor="ex-dept">Exclude my department</label>
                            </div>
                            <div className="filter-checkbox">
                                <input type="checkbox" id="ex-school" checked={excludeSchool} onChange={e => setExcludeSchool(e.target.checked)} />
                                <label htmlFor="ex-school">Exclude my school</label>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Results Column */}
                <div className="results-col">
                    {profile && (
                        <div className="user-hero-card" onClick={() => handleOpenProfile(profile.n_id, true)}>
                            <div className="hero-avatar">
                                <UserCircle size={40} />
                            </div>
                            <div className="hero-info">
                                <h2>{profile.name}</h2>
                                <p>{profile.department} | {profile.school}</p>
                            </div>
                            <div className="hero-hint">View your profile &rarr;</div>
                        </div>
                    )}

                    <div className="neighbours-grid">
                        {loading ? (
                            <div className="loading-state">Finding neighbours...</div>
                        ) : neighbours.length > 0 ? (
                            neighbours.map((n, i) => (
                                <div key={n.n_id} className="neighbour-card" onClick={() => handleOpenProfile(n.n_id)}>
                                    <div className="card-rank">#{i + 1}</div>
                                    <div className="card-main">
                                        <h3 className="neighbour-name">{n.name}</h3>
                                        <p className="neighbour-dept">{n.department}</p>
                                        <p className="neighbour-school">{n.school}</p>
                                    </div>
                                    <div className="card-sim">
                                        <div className="sim-percent">{Math.round(n.similarity * 100)}%</div>
                                        <div className="sim-track">
                                            <div className="sim-bar" style={{ width: `${n.similarity * 100}%` }}></div>
                                        </div>
                                    </div>
                                </div>
                            ))
                        ) : selectedId ? (
                            <div className="empty-state">No neighbours found with current filters.</div>
                        ) : (
                            <div className="empty-state">Search for yourself or a colleague to begin.</div>
                        )}
                    </div>
                </div>
            </div>

            {/* Methodology Modal */}
            {showMethodology && (
                <div className="modal-overlay" onClick={() => setShowMethodology(false)}>
                    <div className="modal-content methodology" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>How Similarity is Computed</h2>
                            <button className="close-btn" onClick={() => setShowMethodology(false)}><X size={24} /></button>
                        </div>
                        <div className="modal-body">
                            <div className="method-step">
                                <div className="step-num">1</div>
                                <div>
                                    <h4>Extract Abstracts</h4>
                                    <p>We analyze publication abstracts from 2019 onwards for every researcher.</p>
                                </div>
                            </div>
                            <div className="method-step">
                                <div className="step-num">2</div>
                                <div>
                                    <h4>Semantic Embedding</h4>
                                    <p>Abstracts are converted into high-dimensional vectors (OpenAI Large-3) that capture research meaning.</p>
                                </div>
                            </div>
                            <div className="method-step">
                                <div className="step-num">3</div>
                                <div>
                                    <h4>Researcher Centroids</h4>
                                    <p>A researcher's "research identity" is the mean vector of all their publication embeddings.</p>
                                </div>
                            </div>
                            <div className="method-step">
                                <div className="step-num">4</div>
                                <div>
                                    <h4>Cosine Similarity</h4>
                                    <p>We measure the distance between centroids in semantic space. High similarity means overlapping research themes.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Profile/Matches Modal */}
            {modalProfile && (
                <div
                    className="modal-overlay"
                    onClick={() => setModalProfile(null)}
                >
                    <div className="modal-content profile-modal" onClick={e => e.stopPropagation()}>
                        {/* Compact dark blue hero header — same style as user-hero-card */}
                        <div className="profile-hero-header">
                            <div className="profile-hero-info">
                                <h2>{modalProfile.name}</h2>
                                <p className="profile-hero-dept">{modalProfile.department} · {modalProfile.school}</p>
                                {modalProfile.email && (
                                    <a href={`mailto:${modalProfile.email}`} className="profile-hero-email">
                                        <Mail size={13} /> {modalProfile.email}
                                    </a>
                                )}
                            </div>
                            <button className="close-btn close-btn-light" onClick={() => setModalProfile(null)}><X size={22} /></button>
                        </div>

                        <div className="modal-body">
                            {modalProfile.research_area && (
                                <div className="profile-section">
                                    <h4 className="section-label">Research Identity</h4>
                                    <p className="summary-text" style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                                        {modalProfile.research_area}
                                    </p>
                                </div>
                            )}

                            {modalProfile.one_line_summary && (
                                <div className="profile-section">
                                    <h4 className="section-label">Profile Summary</h4>
                                    <p className="summary-text">{modalProfile.one_line_summary}</p>
                                </div>
                            )}

                            {modalProfile.subfields && (
                                <div className="profile-section">
                                    <h4 className="section-label">Subfields & Expertise</h4>
                                    <div className="topic-tags">
                                        {modalProfile.subfields.split('|').map(t => <span key={t} className="topic-tag" style={{ background: 'var(--color-primary-light)', color: 'var(--color-primary)' }}>{t.trim()}</span>)}
                                    </div>
                                </div>
                            )}

                            {modalProfile.topics && (
                                <div className="profile-section">
                                    <h4 className="section-label">Topic Domains</h4>
                                    <div className="topic-tags">
                                        {modalProfile.topics.split('|').map(t => <span key={t} className="topic-tag">{t.trim()}</span>)}
                                    </div>
                                </div>
                            )}

                            <div className="profile-section">
                                <h4 className="section-label">
                                    {isOwnProfile ? "Your Recent Publications" : "Top Matches for Your Research"}
                                </h4>
                                <div className="pub-list">
                                    {Array.isArray(modalPubs) && modalPubs.map((pub: any, i) => (
                                        <div key={i} className="pub-card">
                                            <div className="pub-title-row">
                                                {pub.main_url ? (
                                                    <a href={pub.main_url} target="_blank" rel="noreferrer" className="pub-link">
                                                        <span className="pub-title-text">{pub.title}</span>
                                                    </a>
                                                ) : pub.doi ? (
                                                    <a href={pub.doi.startsWith('http') ? pub.doi : `https://doi.org/${pub.doi}`} target="_blank" rel="noreferrer" className="pub-link">
                                                        <span className="pub-title-text">{pub.title}</span>
                                                    </a>
                                                ) : (
                                                    <span className="pub-title-text">{pub.title}</span>
                                                )}
                                                <div className="pub-badges">
                                                    {pub.similarity && <span className="pub-sim">{Math.round(pub.similarity * 100)}%</span>}
                                                    {pub.year && <span className="pub-year">{pub.year}</span>}
                                                </div>
                                            </div>
                                            {pub.authors && (
                                                <p className="pub-authors">{pub.authors}</p>
                                            )}
                                            {pub.abstract && <p className="pub-abstract">{pub.abstract}</p>}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Network;
