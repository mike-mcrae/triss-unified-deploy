import React, { useEffect, useMemo, useState } from 'react';
import { FileText, Mail, X } from 'lucide-react';
import {
  fetchMyPublications,
  fetchReportV3SchoolTopicExperts,
  fetchReportV3SchoolTopics,
  fetchReportV3KeyAreaExperts,
  fetchReportV3KeyAreas,
  fetchReportV3Themes,
  fetchReportV3Overview,
  fetchReportV3School,
  fetchReportV3Schools,
  fetchReportV3Statistics,
  fetchResearcherProfile,
  reportV3PdfUrl,
} from '../api';
import type {
  OwnPublication,
  ReportKeyArea,
  ReportOverviewV3,
  ReportSchoolDetail,
  ReportSchoolSummary,
  ReportMacroTheme,
  ReportTopicExpert,
  ResearcherProfile,
} from '../api';
import '../styles/report.css';
import { useEscapeClose } from '../hooks/useEscapeClose';

type Section = 'overview' | 'themes' | 'statistics' | 'schools' | 'keyareas' | 'methodology';

const SCHOOL_NAME_MAP: Record<string, string> = {
  Business: 'Trinity Business School',
  Education: 'School of Education',
  Law: 'School of Law',
  LSLCS: 'School of Linguistic, Speech and Communication Sciences',
  Psychology: 'School of Psychology',
  RTP: 'School of Religion, Theology and Peace Studies',
  SSP: 'School of Social Sciences and Philosophy',
  SWSP: 'School of Social Work and Social Policy',
};

function displaySchoolName(name?: string) {
  if (!name) return '';
  return SCHOOL_NAME_MAP[name] || name;
}

function truncateText(text: string, maxLen: number): string {
  const clean = (text || '').trim();
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, Math.max(0, maxLen - 1)).trimEnd()}…`;
}

const Report: React.FC = () => {
  const [section, setSection] = useState<Section>('overview');
  const [overview, setOverview] = useState<ReportOverviewV3 | null>(null);
  const [statistics, setStatistics] = useState<any>(null);
  const [schools, setSchools] = useState<ReportSchoolSummary[]>([]);
  const [selectedSchoolKey, setSelectedSchoolKey] = useState<string | null>(null);
  const [selectedSchoolDetail, setSelectedSchoolDetail] = useState<ReportSchoolDetail | null>(null);
  const [selectedSchoolTopics, setSelectedSchoolTopics] = useState<ReportKeyArea[]>([]);
  const [keyAreas, setKeyAreas] = useState<ReportKeyArea[]>([]);
  const [macroThemes, setMacroThemes] = useState<ReportMacroTheme[]>([]);
  const [loading, setLoading] = useState(true);

  const [expertsModalTopic, setExpertsModalTopic] = useState<ReportKeyArea | null>(null);
  const [expertsModalSchool, setExpertsModalSchool] = useState<string | null>(null);
  const [modalTopPx, setModalTopPx] = useState<number>(20);
  const [experts, setExperts] = useState<ReportTopicExpert[]>([]);
  const [expertsLoading, setExpertsLoading] = useState(false);
  const [themeModal, setThemeModal] = useState<ReportMacroTheme | null>(null);
  const [themeArea, setThemeArea] = useState<ReportKeyArea | null>(null);
  const [themeAreaExperts, setThemeAreaExperts] = useState<ReportTopicExpert[]>([]);
  const [themeAreaLoading, setThemeAreaLoading] = useState(false);

  const [profileModal, setProfileModal] = useState<ResearcherProfile | null>(null);
  const [profilePubs, setProfilePubs] = useState<OwnPublication[]>([]);

  useEscapeClose(Boolean(expertsModalTopic), () => { setExpertsModalTopic(null); setExpertsModalSchool(null); });
  useEscapeClose(Boolean(themeModal), () => { setThemeModal(null); setThemeArea(null); });
  useEscapeClose(Boolean(profileModal), () => setProfileModal(null));

  useEffect(() => {
    Promise.all([
      fetchReportV3Overview(),
      fetchReportV3Statistics(),
      fetchReportV3Schools(),
      fetchReportV3KeyAreas(),
      fetchReportV3Themes(),
    ])
      .then(([o, s, sc, k, t]) => {
        setOverview(o);
        setStatistics(s);
        setSchools(sc);
        setKeyAreas(k);
        setMacroThemes(t || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [section, selectedSchoolKey]);

  useEffect(() => {
    if (!selectedSchoolKey) return;
    Promise.all([fetchReportV3School(selectedSchoolKey), fetchReportV3SchoolTopics(selectedSchoolKey)])
      .then(([detail, topics]) => {
        setSelectedSchoolDetail(detail);
        setSelectedSchoolTopics((topics && topics.length ? topics : (detail as any).school_topics) || []);
      })
      .catch(console.error);
  }, [selectedSchoolKey]);

  const summary = overview?.stats;
  const topSchools = useMemo(() => {
    const arr = statistics?.publications_by_school || [];
    return [...arr].sort((a, b) => (b.publications_all || 0) - (a.publications_all || 0));
  }, [statistics]);
  const researchersBySchool = useMemo(() => {
    const arr = statistics?.researchers_by_school || [];
    return [...arr].sort((a, b) => (b.total || 0) - (a.total || 0));
  }, [statistics]);
  const productivity = useMemo(() => {
    const arr = statistics?.school_productivity || [];
    return [...arr].sort((a, b) => (b.avg_pubs_all || 0) - (a.avg_pubs_all || 0));
  }, [statistics]);

  const openExpertsModal = async (topic: ReportKeyArea, schoolKey?: string, anchorY?: number) => {
    setExpertsModalTopic(topic);
    setExpertsModalSchool(schoolKey || null);
    const y = typeof anchorY === 'number' ? anchorY : window.innerHeight * 0.22;
    setModalTopPx(Math.max(12, Math.min(window.innerHeight - 260, y - 80)));
    setExpertsLoading(true);
    try {
      const res = schoolKey
        ? await fetchReportV3SchoolTopicExperts(schoolKey, topic.topic_id, 15)
        : await fetchReportV3KeyAreaExperts(topic.topic_id, 15);
      setExperts(res);
    } catch (err) {
      console.error(err);
      setExperts([]);
    } finally {
      setExpertsLoading(false);
    }
  };

  const openThemeModal = async (theme: ReportMacroTheme, anchorY?: number) => {
    setThemeModal(theme);
    setThemeArea(null);
    setThemeAreaExperts([]);
    const y = typeof anchorY === 'number' ? anchorY : window.innerHeight * 0.22;
    setModalTopPx(Math.max(12, Math.min(window.innerHeight - 260, y - 80)));
  };

  const openThemeArea = async (area: ReportKeyArea) => {
    setThemeArea(area);
    setThemeAreaLoading(true);
    try {
      const res = await fetchReportV3KeyAreaExperts(area.topic_id, 15);
      setThemeAreaExperts(res || []);
    } catch (err) {
      console.error(err);
      setThemeAreaExperts([]);
    } finally {
      setThemeAreaLoading(false);
    }
  };

  const openProfileModal = async (nId: number, anchorY?: number) => {
    try {
      const y = typeof anchorY === 'number' ? anchorY : window.innerHeight * 0.22;
      setModalTopPx(Math.max(12, Math.min(window.innerHeight - 260, y - 80)));
      const [profile, pubs] = await Promise.all([fetchResearcherProfile(nId), fetchMyPublications(nId)]);
      setProfileModal(profile);
      setProfilePubs(Array.isArray(pubs) ? pubs : []);
    } catch (err) {
      console.error(err);
    }
  };

  if (loading) return <div className="page-container fade-in">Loading report...</div>;

  return (
    <div className="page-container fade-in report-page">
      <header className="page-header">
        <h1 className="page-title">TRISS Report</h1>
        <p className="page-subtitle">&nbsp;</p>
      </header>

      <div className="report-nav card">
        {(['overview', 'themes', 'keyareas', 'schools', 'statistics', 'methodology'] as Section[]).map((s) => (
          <button
            key={s}
            type="button"
            className={`report-tab ${section === s ? 'active' : ''}`}
            onClick={() => setSection(s)}
          >
            {s === 'keyareas' ? 'Research' : s === 'themes' ? 'Policy' : s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
        <a className="report-pdf-link" href={reportV3PdfUrl} target="_blank" rel="noreferrer">
          <FileText size={15} /> PDF
        </a>
      </div>

      {section === 'overview' ? (
        <section className="report-section">
          <div className="report-overview-hero">
            <h2>Mapping Research Activity Across TRISS</h2>
            <p>
              Trinity Research in Social Sciences brings together researchers across eight schools,
              producing impactful research with relevance to public policy and society.
            </p>
          </div>

          <div className="stats-grid">
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.total_active_researchers ?? 0}</span><span className="stat-label">Active Researchers</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.num_schools ?? 0}</span><span className="stat-label">Schools</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.num_departments ?? 0}</span><span className="stat-label">Departments</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.total_recent_publications ?? 0}</span><span className="stat-label">Publications (2019+)</span></div></div>
          </div>

          <div className="content-card">
            <h3 className="section-title">About This Report</h3>
            <p className="identity-text">
              This report provides a structured overview of research activity across Trinity Research in Social Sciences (TRISS).
              It combines administrative data, publication metadata, and language model analysis to:
            </p>
            <ul className="report-bullet-list">
              <li>catalogue researchers by department and school,</li>
              <li>summarise individual research agendas in accessible language,</li>
              <li>identify major research themes at departmental, school, and TRISS-wide levels, and</li>
              <li>distil policy relevance of TRISS research activity.</li>
            </ul>
            <p className="identity-text report-aim-text">
              The aim is to present a clear, non-technical account of what TRISS researchers work on and why it matters for policy,
              society, and institutional strategy.
            </p>
            <a className="report-pdf-link" href={reportV3PdfUrl} target="_blank" rel="noreferrer">
              <FileText size={15} /> Download Full Report (PDF)
            </a>
          </div>

          <div className="content-card">
            <h3 className="section-title">Explore The Report</h3>
            <div className="report-explore-grid">
              <button type="button" className="report-explore-card" onClick={() => setSection('keyareas')}>
                <h4>TRISS-Wide Profile</h4>
                <p>Cross-cutting themes, policy relevance, and institutional identity.</p>
              </button>
              <button type="button" className="report-explore-card" onClick={() => setSection('schools')}>
                <h4>Browse By School</h4>
                <p>Explore school themes, departments, and school-specific topic experts.</p>
              </button>
              <button type="button" className="report-explore-card" onClick={() => setSection('statistics')}>
                <h4>Statistics & Trends</h4>
                <p>Compare researcher distribution, publication output, and activity rates.</p>
              </button>
            </div>
          </div>

          <div className="content-card">
            <h3 className="section-title">Institutional Identity</h3>
            <p className="identity-text">{overview?.core_identity || 'No summary available.'}</p>
          </div>
        </section>
      ) : null}

      {section === 'statistics' ? (
        <section className="report-section">
          <div className="stats-grid">
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.total_researchers ?? 0}</span><span className="stat-label">Total Researchers</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.total_active_researchers ?? 0}</span><span className="stat-label">Active Researchers</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.total_publications ?? 0}</span><span className="stat-label">Publications</span></div></div>
            <div className="stat-card"><div className="stat-content"><span className="stat-value">{summary?.percent_recent ?? 0}%</span><span className="stat-label">Recent Share</span></div></div>
          </div>
          <div className="content-card">
            <h3 className="section-title">Researchers by School</h3>
            <div className="report-bars">
              {researchersBySchool.map((s: any) => (
                <div key={`${s.school}-researchers`} className="report-bar-row">
                  <div className="report-bar-label">{s.school}</div>
                  <div className="report-bar-track overlay">
                    <div
                      className="report-bar-fill overall"
                      style={{ width: `${Math.max(2, (s.total / (researchersBySchool[0]?.total || 1)) * 100)}%` }}
                    />
                    <div
                      className="report-bar-fill recent"
                      style={{ width: `${Math.max(2, ((s.active || 0) / (researchersBySchool[0]?.total || 1)) * 100)}%` }}
                    />
                  </div>
                  <div className="report-bar-value">{s.active ?? 0} / {s.total ?? 0}</div>
                </div>
              ))}
            </div>
            <div className="report-legend">
              <span><i className="swatch recent" /> Active (2019+)</span>
              <span><i className="swatch overall" /> All researchers</span>
            </div>
          </div>
          <div className="content-card">
            <h3 className="section-title">Publications by School</h3>
            <div className="report-bars">
              {topSchools.map((s: any) => (
                <div key={s.school} className="report-bar-row">
                  <div className="report-bar-label">{s.school}</div>
                  <div className="report-bar-track overlay">
                    <div
                      className="report-bar-fill overall"
                      style={{ width: `${Math.max(2, (s.publications_all / (topSchools[0]?.publications_all || 1)) * 100)}%` }}
                    />
                    <div
                      className="report-bar-fill recent"
                      style={{ width: `${Math.max(2, ((s.publications_recent || 0) / (topSchools[0]?.publications_all || 1)) * 100)}%` }}
                    />
                  </div>
                  <div className="report-bar-value">{s.publications_recent ?? 0} / {s.publications_all ?? 0}</div>
                </div>
              ))}
            </div>
            <div className="report-legend">
              <span><i className="swatch recent" /> 2019+</span>
              <span><i className="swatch overall" /> All publications</span>
            </div>
            <p className="report-note">Each school shows recent publications versus all publications in the same bar.</p>
          </div>
          <div className="content-card">
            <h3 className="section-title">School Productivity</h3>
            <div className="report-bars">
              {productivity.map((s: any) => (
                <div key={`${s.school}-productivity`} className="report-bar-row">
                  <div className="report-bar-label">{s.school}</div>
                  <div className="report-bar-track overlay">
                    <div
                      className="report-bar-fill overall"
                      style={{ width: `${Math.max(2, (s.avg_pubs_all / (productivity[0]?.avg_pubs_all || 1)) * 100)}%` }}
                    />
                    <div
                      className="report-bar-fill recent"
                      style={{ width: `${Math.max(2, (s.avg_pubs_active / (productivity[0]?.avg_pubs_all || 1)) * 100)}%` }}
                    />
                  </div>
                  <div className="report-bar-value">{s.avg_pubs_active} / {s.avg_pubs_all}</div>
                </div>
              ))}
            </div>
            <div className="report-legend">
              <span><i className="swatch recent" /> Active (2019+ pubs / active researchers)</span>
              <span><i className="swatch overall" /> Overall (all pubs / all researchers)</span>
            </div>
            <p className="report-note">Note: Productivity levels are not directly comparable across schools due to disciplinary publication norms.</p>
          </div>
          <div className="content-card">
            <h3 className="section-title">School Activity Rate</h3>
            <div className="report-bars">
              {productivity
                .slice()
                .sort((a: any, b: any) => (b.share_active || 0) - (a.share_active || 0))
                .map((s: any) => (
                  <div key={`${s.school}-active`} className="report-bar-row">
                    <div className="report-bar-label">{s.school}</div>
                    <div className="report-bar-track">
                      <div className="report-bar-fill" style={{ width: `${Math.max(1, s.share_active || 0)}%` }} />
                    </div>
                    <div className="report-bar-value">{s.share_active}%</div>
                  </div>
                ))}
            </div>
            <p className="report-note">Note: Activity rate = share of researchers with at least one publication since 2019.</p>
          </div>
        </section>
      ) : null}

      {section === 'schools' ? (
        <section className="report-section">
          {!selectedSchoolKey ? (
            <div className="report-schools-grid">
              {schools.map((school) => (
                <button key={school.key} type="button" className="report-school-card" onClick={() => setSelectedSchoolKey(school.key)}>
                  <h3>{displaySchoolName(school.name)}</h3>
                  <p>{school.researchers} researchers</p>
                  <p>{school.publications_recent ?? 0} publications (2019+)</p>
                </button>
              ))}
            </div>
          ) : (
            <div className="content-card">
              <div className="report-inline-header">
                <h3 className="section-title">{displaySchoolName(selectedSchoolDetail?.name || selectedSchoolKey || '')}</h3>
                <button type="button" className="method-btn" onClick={() => { setSelectedSchoolKey(null); setSelectedSchoolDetail(null); }}>
                  Back
                </button>
              </div>
              <div className="stats-grid" style={{ marginBottom: '0.8rem' }}>
                <div className="stat-card"><div className="stat-content"><span className="stat-value">{selectedSchoolDetail?.researchers ?? 0}</span><span className="stat-label">Researchers</span></div></div>
                <div className="stat-card"><div className="stat-content"><span className="stat-value">{selectedSchoolDetail?.publications_recent ?? 0}</span><span className="stat-label">Publications (2019+)</span></div></div>
              </div>
              <h4 className="section-label">School Topic Areas (Click for Experts)</h4>
              <div className="report-key-grid">
                {selectedSchoolTopics.map((topic) => (
                  <button
                    key={`${selectedSchoolKey}-${topic.topic_id}`}
                    type="button"
                    className="theme-card report-topic-card"
                    onClick={(e) => openExpertsModal(topic, selectedSchoolKey, e.clientY)}
                  >
                    <h4 className="theme-name">{topic.theme_name}</h4>
                    <p className="theme-description">{topic.description}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      ) : null}

      {section === 'keyareas' ? (
        <section className="report-section">
          <div className="content-card">
            <h3 className="section-title">Key Areas of TRISS Research</h3>
            <p className="page-subtitle report-subcopy">Click an area to view top semantically aligned researchers.</p>
            <div className="report-key-grid">
              {keyAreas.map((topic) => (
                <button
                  key={topic.topic_id}
                  type="button"
                  className="theme-card report-topic-card"
                  onClick={(e) => openExpertsModal(topic, undefined, e.clientY)}
                >
                  <h4 className="theme-name">{topic.theme_name}</h4>
                  <p className="theme-description">{topic.description}</p>
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      {section === 'themes' ? (
        <section className="report-section">
          <div className="content-card">
            <h3 className="section-title">Policy Domains</h3>
            <p className="page-subtitle report-subcopy">
              Click a policy domain to view linked research areas and top semantically aligned researchers.
            </p>
            <div className="report-key-grid">
              {macroThemes.map((theme) => (
                <button
                  key={theme.theme_id}
                  type="button"
                  className="theme-card report-topic-card"
                  onClick={(e) => openThemeModal(theme, e.clientY)}
                >
                  <h4 className="theme-name">{theme.title}</h4>
                  <p className="theme-description">{truncateText(theme.description || '', 210)}</p>
                  <p className="report-theme-meta">{theme.n_researchers} researchers</p>
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      {section === 'methodology' ? (
        <section className="report-section">
          <div className="report-method-hero">
            <p>
              This report combines administrative data, publication metadata, semantic embeddings, clustering, and
              LLM-assisted synthesis to catalogue researchers, summarise research agendas, identify themes, and distil
              policy relevance. The pipeline is reproducible, fault-tolerant, and auditable.
            </p>
          </div>

          <div className="content-card">
            <h3 className="section-title">Data Pipeline</h3>
            <div className="report-pipeline-grid report-pipeline-grid-4">
              <div className="report-pipeline-card blue">
                <div className="report-pipeline-step">Step 1</div>
                <div className="report-pipeline-icon">📥</div>
                <h4>Data Collection</h4>
                <ul>
                  <li>Scrape researcher profiles from RSS</li>
                  <li>Extract publication records</li>
                  <li>{summary?.num_schools ?? 0} schools, {summary?.total_active_researchers ?? 0} active researchers</li>
                  <li>{summary?.total_recent_publications ?? 0} publications (2019+)</li>
                </ul>
              </div>
              <div className="report-pipeline-card blue">
                <div className="report-pipeline-step">Step 2</div>
                <div className="report-pipeline-icon">🔗</div>
                <h4>Abstract Matching</h4>
                <ul>
                  <li>Match via Crossref API</li>
                  <li>Supplement with OpenAlex</li>
                  <li>Web scrape remaining</li>
                </ul>
              </div>
              <div className="report-pipeline-card blue">
                <div className="report-pipeline-step">Step 3</div>
                <div className="report-pipeline-icon">🧮</div>
                <h4>Semantic Embedding</h4>
                <ul>
                  <li>Embed abstracts in vector space</li>
                  <li>Compute similarity matrices</li>
                  <li>Identify nearest neighbours</li>
                </ul>
              </div>
              <div className="report-pipeline-card green">
                <div className="report-pipeline-step">Step 4</div>
                <div className="report-pipeline-icon">🤖</div>
                <h4>LLM Analysis</h4>
                <ul>
                  <li>Extract themes per researcher</li>
                  <li>Aggregate to unit level</li>
                  <li>Synthesise institution profile</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="content-card">
            <h3 className="section-title">LLM-Based Analysis: Three-Level Aggregation</h3>
            <div className="report-llm-grid">
              <div className="report-llm-card blue">
                <div className="report-llm-title"><span>Prompt 1</span><h4>Researcher Level</h4></div>
                <p>Given RSS profile text and recent publication abstracts, infer research themes, methods, and policy relevance.</p>
                <div className="report-llm-io">Input: Profile + Abstracts<br />Output: Structured JSON summary</div>
              </div>
              <div className="report-llm-card blue-light">
                <div className="report-llm-title"><span>Prompt 2</span><h4>Unit Level</h4></div>
                <p>Synthesise researcher summaries into department/school profiles, identifying dominant themes and patterns.</p>
                <div className="report-llm-io">Input: Researcher summaries<br />Output: Unit-level profile</div>
              </div>
              <div className="report-llm-card green">
                <div className="report-llm-title"><span>Prompt 3</span><h4>Institution Level</h4></div>
                <p>Integrate school-level analyses into a single TRISS-wide synthesis of cross-cutting themes and identity.</p>
                <div className="report-llm-io">Input: School summaries<br />Output: TRISS profile</div>
              </div>
            </div>
          </div>

          <div className="content-card">
            <h3 className="section-title">Data Sources</h3>
            <div className="report-sources-grid">
              <div className="report-source"><div>🏛️</div><div><h4>Trinity RSS</h4><p>Official researcher profiles and publication lists</p></div></div>
              <div className="report-source"><div>📚</div><div><h4>Crossref API</h4><p>DOIs, abstracts, and bibliographic metadata</p></div></div>
              <div className="report-source"><div>🔬</div><div><h4>OpenAlex</h4><p>Supplementary abstract retrieval</p></div></div>
              <div className="report-source"><div>🌐</div><div><h4>Publisher Websites</h4><p>Direct scraping for remaining abstracts</p></div></div>
            </div>
          </div>
        </section>
      ) : null}

      {expertsModalTopic ? (
        <div className="modal-overlay report-floating-overlay" onClick={() => { setExpertsModalTopic(null); setExpertsModalSchool(null); }}>
          <div className="modal-content profile-modal report-floating-modal" style={{ marginTop: `${modalTopPx}px` }} onClick={(e) => e.stopPropagation()}>
            <div className="profile-hero-header">
              <div className="profile-hero-info">
                <h2>{expertsModalTopic.theme_name}</h2>
                <p className="profile-hero-dept">{expertsModalSchool ? displaySchoolName(expertsModalSchool) : 'TRISS Key Area'}</p>
              </div>
              <button className="close-btn close-btn-light" onClick={() => { setExpertsModalTopic(null); setExpertsModalSchool(null); }}>
                <X size={22} />
              </button>
            </div>
            <div className="modal-body">
              <p className="theme-description">{expertsModalTopic.description}</p>
              <h4 className="section-label">Top Semantic Experts</h4>
              {expertsLoading ? (
                <div className="loading-state">Loading experts...</div>
              ) : (
                <div className="expert-list">
                  {experts.map((expert, idx) => (
                    <div key={`${expert.n_id}-${idx}`} className="neighbour-card expert-card" onClick={(e) => openProfileModal(expert.n_id, e.clientY)}>
                      <div className="card-rank">#{idx + 1}</div>
                      <div className="card-main">
                        <h3 className="neighbour-name">{expert.name}</h3>
                        <p className="neighbour-dept">{expert.department}</p>
                        <p className="neighbour-school">{expert.school}</p>
                      </div>
                      <div className="card-sim">
                        <div className="sim-percent">{Math.round(expert.similarity * 100)}%</div>
                        <div className="sim-track"><div className="sim-bar" style={{ width: `${Math.max(1, expert.similarity * 100)}%` }} /></div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {profileModal ? (
        <div className="modal-overlay report-floating-overlay report-overlay-top" onClick={() => setProfileModal(null)}>
          <div className="modal-content profile-modal report-floating-modal" style={{ marginTop: `${modalTopPx}px` }} onClick={(e) => e.stopPropagation()}>
            <div className="profile-hero-header">
              <div className="profile-hero-info">
                <h2>{profileModal.name}</h2>
                <p className="profile-hero-dept">{profileModal.department} · {profileModal.school}</p>
                {profileModal.email ? (
                  <a href={`mailto:${profileModal.email}`} className="profile-hero-email">
                    <Mail size={13} /> {profileModal.email}
                  </a>
                ) : null}
              </div>
              <button className="close-btn close-btn-light" onClick={() => setProfileModal(null)}>
                <X size={22} />
              </button>
            </div>
            <div className="modal-body">
              {profileModal.research_area ? (
                <div className="profile-section">
                  <h4 className="section-label">Research Identity</h4>
                  <p className="summary-text" style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                    {profileModal.research_area}
                  </p>
                </div>
              ) : null}

              {profileModal.one_line_summary ? (
                <div className="profile-section">
                  <h4 className="section-label">Profile Summary</h4>
                  <p className="summary-text">{profileModal.one_line_summary}</p>
                </div>
              ) : null}

              <div className="profile-section">
                <h4 className="section-label">Recent Publications</h4>
                <div className="pub-list">
                  {profilePubs.map((pub, i) => (
                    <div key={`${pub.title}-${i}`} className="pub-card">
                      <div className="pub-title-row">
                        {pub.main_url ? (
                          <a href={pub.main_url} target="_blank" rel="noreferrer" className="pub-link">
                            <span className="pub-title-text">{pub.title}</span>
                          </a>
                        ) : (
                          <span className="pub-title-text">{pub.title}</span>
                        )}
                        <div className="pub-badges">{pub.year ? <span className="pub-year">{pub.year}</span> : null}</div>
                      </div>
                      {pub.abstract ? <p className="pub-abstract">{pub.abstract}</p> : null}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {themeModal ? (
        <div className="modal-overlay report-floating-overlay" onClick={() => { setThemeModal(null); setThemeArea(null); }}>
          <div className="modal-content profile-modal report-floating-modal" style={{ marginTop: `${modalTopPx}px` }} onClick={(e) => e.stopPropagation()}>
            <div className="profile-hero-header">
              <div className="profile-hero-info">
                <h2>{themeModal.title}</h2>
                <p className="profile-hero-dept">Overarching TRISS Theme</p>
              </div>
              <button className="close-btn close-btn-light" onClick={() => { setThemeModal(null); setThemeArea(null); }}>
                <X size={22} />
              </button>
            </div>
            <div className="modal-body">
              <p className="theme-description">{themeModal.description}</p>
              {themeModal.policy_relevance ? (
                <p className="report-note"><strong>Research informs:</strong> {themeModal.policy_relevance}</p>
              ) : null}
              {themeModal.aligned_research_areas?.length ? (
                <>
                  <h4 className="section-label">Aligned Research Areas</h4>
                  <p className="report-note">Select a research area below to view top aligned researchers.</p>
                  <div className="report-theme-chip-list">
                    {themeModal.aligned_research_areas.map((area) => (
                      <button
                        key={`theme-${themeModal.theme_id}-topic-${area.topic_id}`}
                        type="button"
                        className="report-theme-chip"
                        onClick={() => openThemeArea(area)}
                      >
                        {area.theme_name}
                      </button>
                    ))}
                  </div>
                </>
              ) : null}
              {themeArea ? (
                <>
                  <h4 className="section-label">{themeArea.theme_name}</h4>
                  <p className="theme-description">{themeArea.description}</p>
                  <h4 className="section-label">Top Researchers</h4>
                  {themeAreaLoading ? (
                    <div className="loading-state">Loading researchers...</div>
                  ) : (
                    <div className="expert-list">
                      {themeAreaExperts.map((expert, idx) => (
                        <div key={`${expert.n_id}-${idx}`} className="neighbour-card expert-card" onClick={(e) => openProfileModal(expert.n_id, e.clientY)}>
                          <div className="card-rank">#{idx + 1}</div>
                          <div className="card-main">
                            <h3 className="neighbour-name">{expert.name}</h3>
                            <p className="neighbour-dept">{expert.department}</p>
                            <p className="neighbour-school">{expert.school}</p>
                          </div>
                          <div className="card-sim">
                            <div className="sim-percent">{Math.round(expert.similarity * 100)}%</div>
                            <div className="sim-track"><div className="sim-bar" style={{ width: `${Math.max(1, expert.similarity * 100)}%` }} /></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default Report;
