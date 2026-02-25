import React, { useEffect, useMemo, useState } from 'react';
import { ChevronDown, Mail, Search, X } from 'lucide-react';
import {
  fetchExpertSearchFilters,
  fetchExpertThemeOptions,
  fetchQueryRankedPublications,
  fetchResearcherProfile,
  runExpertSearch,
  runExpertThemeSearch,
} from '../api';
import type { ExpertPublicationResult, ExpertSearchFilters, ExpertSearchResponse, ExpertThemeOption, ResearcherProfile } from '../api';
import '../styles/expert.css';
import { useEscapeClose } from '../hooks/useEscapeClose';

type LastSearchState =
  | { mode: 'query'; query: string }
  | { mode: 'theme'; optionId: string; label: string };

const ExpertFinder: React.FC = () => {
  const [query, setQuery] = useState('');
  const [themeOptions, setThemeOptions] = useState<ExpertThemeOption[]>([]);
  const [submittedQuery, setSubmittedQuery] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showAllSuggestions, setShowAllSuggestions] = useState(false);
  const [results, setResults] = useState<ExpertSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchFilters, setSearchFilters] = useState<ExpertSearchFilters>({ schools: [], departments_by_school: {} });
  const [selectedSchool, setSelectedSchool] = useState('');
  const [selectedDepartment, setSelectedDepartment] = useState('');
  const [lastSearch, setLastSearch] = useState<LastSearchState | null>(null);
  const [modalProfile, setModalProfile] = useState<ResearcherProfile | null>(null);
  const [modalPubs, setModalPubs] = useState<ExpertPublicationResult[]>([]);
  const [modalLoading, setModalLoading] = useState(false);

  useEscapeClose(Boolean(modalProfile), () => setModalProfile(null));

  useEffect(() => {
    fetchExpertThemeOptions().then(setThemeOptions).catch(console.error);
    fetchExpertSearchFilters().then(setSearchFilters).catch(console.error);
  }, []);

  const themeOptionByQuery = useMemo(() => {
    const map = new Map<string, ExpertThemeOption>();
    const norm = (s: string) => s.trim().toLowerCase();
    const coreLabel = (label: string) => {
      const stripped = label.replace(/^TRISS - /, '').trim();
      const idx = stripped.indexOf(' - ');
      return idx >= 0 ? stripped.slice(idx + 3).trim() : stripped;
    };
    for (const opt of themeOptions) {
      const full = (opt.label || '').trim();
      if (!full) continue;
      if (!map.has(norm(full))) map.set(norm(full), opt);
      const trissShort = full.replace(/^TRISS - /, '').trim();
      if (trissShort && !map.has(norm(trissShort))) map.set(norm(trissShort), opt);
      const core = coreLabel(full);
      if (core && !map.has(norm(core))) map.set(norm(core), opt);
    }
    return map;
  }, [themeOptions]);

  const predictiveQuerySuggestions = useMemo(() => {
    const values = new Set<string>();
    const coreLabel = (label: string) => {
      const stripped = label.replace(/^TRISS - /, '').trim();
      const idx = stripped.indexOf(' - ');
      return idx >= 0 ? stripped.slice(idx + 3).trim() : stripped;
    };
    for (const opt of themeOptions) {
      const label = (opt.label || '').trim();
      if (!label) continue;
      const core = coreLabel(label);
      if (core) values.add(core);
    }
    return Array.from(values).filter(Boolean).sort((a, b) => a.localeCompare(b));
  }, [themeOptions]);

  const filteredSuggestions = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 3) return [];
    return predictiveQuerySuggestions
      .filter((s) => s.toLowerCase().includes(q))
      .slice(0, 12);
  }, [predictiveQuerySuggestions, query]);

  const displayedSuggestions = useMemo(() => {
    if (showAllSuggestions) return predictiveQuerySuggestions;
    return filteredSuggestions;
  }, [showAllSuggestions, predictiveQuerySuggestions, filteredSuggestions]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const clean = query.trim();
    if (!clean) return;
    const matchedTheme = themeOptionByQuery.get(clean.toLowerCase());
    setLoading(true);
    setError(null);
    setSubmittedQuery(clean);
    try {
      let payload: ExpertSearchResponse;
      if (matchedTheme) {
        setSubmittedQuery(matchedTheme.label || clean);
        payload = await runExpertThemeSearch(
          matchedTheme.option_id,
          12,
          20,
          selectedSchool || undefined,
          selectedDepartment || undefined
        );
        setLastSearch({ mode: 'theme', optionId: matchedTheme.option_id, label: matchedTheme.label || clean });
      } else {
        payload = await runExpertSearch(
          clean,
          12,
          20,
          selectedSchool || undefined,
          selectedDepartment || undefined
        );
        setLastSearch({ mode: 'query', query: clean });
      }
      setResults(payload);
    } catch (err: any) {
      setError(err?.message || 'Search failed.');
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!lastSearch) return;
    let isCancelled = false;
    const rerun = async () => {
      setLoading(true);
      setError(null);
      try {
        let payload: ExpertSearchResponse;
        if (lastSearch.mode === 'theme') {
          payload = await runExpertThemeSearch(
            lastSearch.optionId,
            12,
            20,
            selectedSchool || undefined,
            selectedDepartment || undefined
          );
          if (!isCancelled) {
            setSubmittedQuery(lastSearch.label);
          }
        } else {
          payload = await runExpertSearch(
            lastSearch.query,
            12,
            20,
            selectedSchool || undefined,
            selectedDepartment || undefined
          );
          if (!isCancelled) {
            setSubmittedQuery(lastSearch.query);
          }
        }
        if (!isCancelled) setResults(payload);
      } catch (err: any) {
        if (!isCancelled) {
          setError(err?.message || 'Search failed.');
          setResults(null);
        }
      } finally {
        if (!isCancelled) setLoading(false);
      }
    };
    rerun();
    return () => {
      isCancelled = true;
    };
  }, [selectedSchool, selectedDepartment]);

  const departmentOptions = useMemo(() => {
    if (!selectedSchool) {
      const all = new Set<string>();
      Object.values(searchFilters.departments_by_school || {}).forEach((arr) => arr.forEach((d) => all.add(d)));
      return Array.from(all).sort((a, b) => a.localeCompare(b));
    }
    return searchFilters.departments_by_school[selectedSchool] || [];
  }, [searchFilters, selectedSchool]);

  const openProfileModal = async (nId: number) => {
    try {
      setModalLoading(true);
      const queryForRanking = (submittedQuery || query).trim();
      const [profile, pubs] = await Promise.all([
        fetchResearcherProfile(nId),
        fetchQueryRankedPublications(nId, queryForRanking, 20),
      ]);
      setModalProfile(profile);
      setModalPubs(Array.isArray(pubs) ? pubs : []);
    } catch (err) {
      console.error('Failed to open profile modal:', err);
      setModalProfile(null);
      setModalPubs([]);
    } finally {
      setModalLoading(false);
    }
  };

  return (
    <div className="page-container fade-in">
      <header className="page-header">
        <h1 className="page-title">Expert Finder</h1>
        <p className="page-subtitle">
          Use either option below: enter a free-text semantic query, or select a research theme from the dropdown.
        </p>
      </header>

      <form className="expert-search-form card" onSubmit={onSubmit}>
        <div className="expert-query-row">
          <div className="expert-search-suggest">
            <div className="search-box">
              <Search className="search-icon" size={18} />
              <input
                className="search-input"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setShowAllSuggestions(false);
                }}
                onFocus={() => setShowSuggestions(true)}
                onBlur={() => setTimeout(() => {
                  setShowSuggestions(false);
                  setShowAllSuggestions(false);
                }, 120)}
                placeholder="Start typing - choose from list or type your own query"
              />
              <button
                type="button"
                className="expert-suggest-toggle"
                aria-label="Show all suggestions"
                onMouseDown={(e) => {
                  e.preventDefault();
                  setShowSuggestions(true);
                  setShowAllSuggestions((prev) => !prev);
                }}
              >
                <ChevronDown size={16} />
              </button>
            </div>
            {showSuggestions && displayedSuggestions.length > 0 ? (
              <div className="search-dropdown">
                {displayedSuggestions.map((value) => (
                  <div
                    key={value}
                    className="dropdown-item"
                    onMouseDown={(e) => {
                      e.preventDefault();
                      setQuery(value);
                      setShowSuggestions(false);
                      setShowAllSuggestions(false);
                    }}
                  >
                    <div className="item-name">{value}</div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
          <button className="method-btn expert-submit-btn" type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Find Experts'}
          </button>
        </div>
        <div className="expert-filter-row">
          <select
            value={selectedSchool}
            onChange={(e) => {
              setSelectedSchool(e.target.value);
              setSelectedDepartment('');
            }}
          >
            <option value="">All schools</option>
            {searchFilters.schools.map((school) => (
              <option key={school} value={school}>{school}</option>
            ))}
          </select>
          <select
            value={selectedDepartment}
            onChange={(e) => setSelectedDepartment(e.target.value)}
          >
            <option value="">All departments</option>
            {departmentOptions.map((dept) => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </select>
        </div>
      </form>

      {error ? <div className="empty-state expert-error">{error}</div> : null}

      {!loading && !results && !error ? (
        <div className="empty-state">Enter a phrase to retrieve aligned researchers and publications.</div>
      ) : null}

      {results ? (
        <div className="expert-layout">
          <section className="card expert-col">
            <h3 className="card-title">Top Researchers</h3>
            <div className="expert-list">
              {results.top_researchers.map((r, idx) => (
                <div
                  key={r.n_id}
                  className="neighbour-card expert-card"
                  onClick={() => openProfileModal(r.n_id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      openProfileModal(r.n_id);
                    }
                  }}
                >
                  <div className="card-rank">#{idx + 1}</div>
                  <div className="card-main">
                    <h3 className="neighbour-name">{r.name}</h3>
                    <p className="neighbour-dept">{r.department || 'Department unavailable'}</p>
                    <p className="neighbour-school">{r.school || 'School unavailable'}</p>
                    <span className="expert-link-btn">View Profile</span>
                  </div>
                  <div className="card-sim">
                    <div className="sim-percent">{Math.round(r.similarity * 100)}%</div>
                    <div className="sim-track">
                      <div className="sim-bar" style={{ width: `${Math.max(0, Math.min(100, r.similarity * 100))}%` }} />
                    </div>
                  </div>
                </div>
              ))}
              {!results.top_researchers.length ? <div className="empty-state">No researcher matches found.</div> : null}
            </div>
          </section>

          <section className="card expert-col">
            <h3 className="card-title">Top Publications</h3>
            <div className="pub-list">
              {results.top_publications.map((pub, idx) => (
                <div
                  key={`${pub.article_id}-${idx}`}
                  className={`pub-card ${pub.main_url ? 'expert-pub-clickable' : ''}`}
                  onClick={() => {
                    if (pub.main_url) window.open(pub.main_url, '_blank', 'noopener,noreferrer');
                  }}
                  role={pub.main_url ? 'link' : undefined}
                  tabIndex={pub.main_url ? 0 : -1}
                  onKeyDown={(e) => {
                    if (!pub.main_url) return;
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      window.open(pub.main_url, '_blank', 'noopener,noreferrer');
                    }
                  }}
                >
                  <div className="pub-title-row">
                    {pub.main_url ? (
                      <a href={pub.main_url} target="_blank" rel="noreferrer" className="pub-link">
                        <span className="pub-title-text">{pub.title}</span>
                      </a>
                    ) : (
                      <span className="pub-title-text">{pub.title}</span>
                    )}
                    <div className="pub-badges">
                      <span className="pub-sim">{Math.round(pub.similarity * 100)}%</span>
                    </div>
                  </div>
                  <p className="expert-byline">
                    <button
                      type="button"
                      className="expert-name-link"
                      onClick={(e) => {
                        e.stopPropagation();
                        openProfileModal(pub.n_id);
                      }}
                    >
                      {pub.researcher_name || `Researcher ${pub.n_id}`}
                    </button>
                    {pub.department ? (
                      <span className="expert-byline-meta">
                        {' '}| {pub.department}
                      </span>
                    ) : null}
                  </p>
                  {pub.abstract_snippet ? <p className="pub-abstract">{pub.abstract_snippet}</p> : null}
                </div>
              ))}
              {!results.top_publications.length ? <div className="empty-state">No publication matches found.</div> : null}
            </div>
          </section>
        </div>
      ) : null}

      {modalProfile ? (
        <div className="modal-overlay expert-profile-overlay" onClick={() => setModalProfile(null)}>
          <div className="modal-content profile-modal" onClick={(e) => e.stopPropagation()}>
            <div className="profile-hero-header">
              <div className="profile-hero-info">
                <h2>{modalProfile.name}</h2>
                <p className="profile-hero-dept">{modalProfile.department} · {modalProfile.school}</p>
                {modalProfile.email ? (
                  <a href={`mailto:${modalProfile.email}`} className="profile-hero-email">
                    <Mail size={13} /> {modalProfile.email}
                  </a>
                ) : null}
              </div>
              <button className="close-btn close-btn-light" onClick={() => setModalProfile(null)}>
                <X size={22} />
              </button>
            </div>

            <div className="modal-body">
              {modalProfile.research_area ? (
                <div className="profile-section">
                  <h4 className="section-label">Research Identity</h4>
                  <p className="summary-text" style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                    {modalProfile.research_area}
                  </p>
                </div>
              ) : null}

              {modalProfile.one_line_summary ? (
                <div className="profile-section">
                  <h4 className="section-label">Profile Summary</h4>
                  <p className="summary-text">{modalProfile.one_line_summary}</p>
                </div>
              ) : null}

              <div className="profile-section">
                <h4 className="section-label">Publications Most Aligned With: "{submittedQuery}"</h4>
                {modalLoading ? (
                  <div className="loading-state">Loading publications...</div>
                ) : (
                  <div className="pub-list">
                    {modalPubs.map((pub, i) => (
                      <div key={`${pub.article_id}-${i}`} className="pub-card">
                        <div className="pub-title-row">
                          {pub.main_url ? (
                            <a href={pub.main_url} target="_blank" rel="noreferrer" className="pub-link">
                              <span className="pub-title-text">{pub.title}</span>
                            </a>
                          ) : (
                            <span className="pub-title-text">{pub.title}</span>
                          )}
                          <div className="pub-badges">
                            <span className="pub-sim">{Math.round(pub.similarity * 100)}%</span>
                          </div>
                        </div>
                        {pub.abstract_snippet ? <p className="pub-abstract">{pub.abstract_snippet}</p> : null}
                      </div>
                    ))}
                    {!modalPubs.length ? (
                      <div className="empty-state">No query-ranked publications found for this researcher.</div>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default ExpertFinder;
