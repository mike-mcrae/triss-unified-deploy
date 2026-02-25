import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { fetchMyPublications, fetchQueryRankedPublications, fetchResearcherProfile } from '../api';
import type { ExpertPublicationResult, OwnPublication, ResearcherProfile } from '../api';
import '../styles/expert.css';

type ProfilePublication = (OwnPublication & { similarity?: number }) | ExpertPublicationResult;

const Profile: React.FC = () => {
  const [params] = useSearchParams();
  const nId = Number(params.get('n_id') || '');
  const query = (params.get('query') || '').trim();

  const [profile, setProfile] = useState<ResearcherProfile | null>(null);
  const [publications, setPublications] = useState<ProfilePublication[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(nId) || nId <= 0) {
      setError('Invalid researcher id.');
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    Promise.all([
      fetchResearcherProfile(nId),
      query ? fetchQueryRankedPublications(nId, query, 20) : fetchMyPublications(nId),
    ])
      .then(([p, pubs]) => {
        setProfile(p);
        setPublications(Array.isArray(pubs) ? pubs : []);
      })
      .catch((err: any) => setError(err?.message || 'Failed to load profile.'))
      .finally(() => setLoading(false));
  }, [nId, query]);

  if (loading) {
    return <div className="page-container"><div className="loading-state">Loading profile...</div></div>;
  }

  if (error || !profile) {
    return (
      <div className="page-container">
        <div className="empty-state">{error || 'Profile not found.'}</div>
      </div>
    );
  }

  return (
    <div className="page-container fade-in">
      <header className="page-header">
        <h1 className="page-title">{profile.name}</h1>
        <p className="page-subtitle">{profile.department} · {profile.school}</p>
      </header>

      <div className="expert-profile-actions">
        <Link to="/expert-search" className="method-btn">Back to Expert Finder</Link>
      </div>

      <div className="card expert-profile-card">
        {profile.research_area ? (
          <div className="profile-section">
            <h4 className="section-label">Research Identity</h4>
            <p className="summary-text">{profile.research_area}</p>
          </div>
        ) : null}

        {profile.one_line_summary ? (
          <div className="profile-section">
            <h4 className="section-label">Profile Summary</h4>
            <p className="summary-text">{profile.one_line_summary}</p>
          </div>
        ) : null}

        <div className="profile-section">
          <h4 className="section-label">
            {query ? `Publications Most Aligned With: "${query}"` : 'Recent Publications'}
          </h4>
          <div className="pub-list">
            {publications.map((pub: any, idx) => (
              <div key={`${pub.article_id || pub.title}-${idx}`} className="pub-card">
                <div className="pub-title-row">
                  <span className="pub-title-text">{pub.title}</span>
                  <div className="pub-badges">
                    {typeof pub.similarity === 'number' ? <span className="pub-sim">{Math.round(pub.similarity * 100)}%</span> : null}
                    {pub.year ? <span className="pub-year">{pub.year}</span> : null}
                  </div>
                </div>
                {(pub.abstract_snippet || pub.abstract) ? (
                  <p className="pub-abstract">{pub.abstract_snippet || pub.abstract}</p>
                ) : null}
              </div>
            ))}
            {!publications.length ? (
              <div className="empty-state">
                {query ? 'No query-ranked publications found for this researcher.' : 'No publications found.'}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Profile;
