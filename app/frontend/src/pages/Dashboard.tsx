import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, Compass, Map as MapIcon, SearchCode, Users } from 'lucide-react';
import { fetchReportV3Overview } from '../api';
import type { ReportOverviewV3 } from '../api';
import '../styles/dashboard.css';

const Dashboard: React.FC = () => {
    const [data, setData] = useState<ReportOverviewV3 | null>(null);
    const [loading, setLoading] = useState(true);
    const policyDomains = [
        'Democratic Governance & Regulatory Systems',
        'Economic Policy & Sustainable Development',
        'Health & Mental Health Systems',
        'Social Protection, Welfare & Peacebuilding',
        'Inclusive Education & Integration',
    ];

    useEffect(() => {
        fetchReportV3Overview()
            .then(setData)
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div className="page-container fade-in">Loading dashboard...</div>;
    if (!data) return <div className="page-container fade-in">Error loading data.</div>;

    return (
        <div className="page-container fade-in dashboard-page">
            <header className="page-header dashboard-hero">
                <p className="dashboard-eyebrow">TRISS Unified Research Portal</p>
                <h1 className="page-title">Explore Trinity Social Sciences Research</h1>
                <p className="page-subtitle dashboard-subtitle">
                    Navigate the full TRISS research landscape: report synthesis, thematic map exploration,
                    academic neighbour discovery, and expert finding.
                </p>
            </header>

            <div className="stats-grid dashboard-stats">
                <div className="stat-card">
                    <div className="stat-icon-wrapper blue">
                        <Users size={24} />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{data.stats.total_active_researchers ?? 0}</span>
                        <span className="stat-label">Active Researchers</span>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon-wrapper purple">
                        <BookOpen size={24} />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{data.stats.num_schools}</span>
                        <span className="stat-label">Schools</span>
                    </div>
                </div>
                <div className="stat-card">
                    <div className="stat-icon-wrapper indigo">
                        <Compass size={24} />
                    </div>
                    <div className="stat-content">
                        <span className="stat-value">{data.stats.total_recent_publications ?? 0}</span>
                        <span className="stat-label">Publications (2019+)</span>
                    </div>
                </div>
            </div>

            <section className="content-card dashboard-identity">
                <h2 className="section-title">About TRISS</h2>
                <p className="identity-text dashboard-identity-intro">
                    TRISS (Trinity Research in Social Sciences) is an interdisciplinary research centre at Trinity College Dublin focused on strengthening democratic
                    governance, advancing inclusive social systems, and supporting sustainable economic and technological
                    development. Drawing expertise from Business, Social Sciences, Education, Law, Religion, Psychology, Social
                    Policy, and Linguistics, TRISS addresses complex societal challenges through integrated research across five
                    core policy domains:
                </p>
                <div className="dashboard-domain-grid">
                    {policyDomains.map((domain, index) => (
                        <article key={domain} className="dashboard-domain-card">
                            <span className="dashboard-domain-index">{String(index + 1).padStart(2, '0')}</span>
                            <h3>{domain}</h3>
                        </article>
                    ))}
                </div>
                <p className="identity-text dashboard-identity-impact">
                    TRISS produces analytically rigorous and policy-relevant research that informs institutional reform,
                    regulatory design, and evidence-based public decision-making. Through active engagement with government,
                    regulatory authorities, industry leaders, international organizations, and civil society, TRISS strengthens
                    the link between academic research and real-world policy impact.
                </p>
            </section>

            <section className="dashboard-actions-wrap">
                <h2 className="section-title">What You Can Do Here</h2>
                <div className="dashboard-actions-grid">
                    <Link to="/report" className="dashboard-action-card">
                        <div className="dashboard-action-icon blue">
                            <BookOpen size={20} />
                        </div>
                        <h3>Read the Report</h3>
                        <p>Review the full TRISS synthesis, key areas, schools, statistics, and methodology.</p>
                    </Link>

                    <Link to="/map" className="dashboard-action-card">
                        <div className="dashboard-action-icon orange">
                            <MapIcon size={20} />
                        </div>
                        <h3>Map Research Themes</h3>
                        <p>Explore publication clusters, topic distributions, and school-level thematic structure.</p>
                    </Link>

                    <Link to="/network" className="dashboard-action-card">
                        <div className="dashboard-action-icon green">
                            <Users size={20} />
                        </div>
                        <h3>Find Academic Neighbours</h3>
                        <p>Identify semantically close researchers based on publication embedding similarity.</p>
                    </Link>

                    <Link to="/expert-search" className="dashboard-action-card">
                        <div className="dashboard-action-icon purple">
                            <SearchCode size={20} />
                        </div>
                        <h3>Find Experts</h3>
                        <p>Search by free text or theme terms to retrieve aligned researchers and publications.</p>
                    </Link>
                </div>
            </section>

        </div>
    );
};

export default Dashboard;
