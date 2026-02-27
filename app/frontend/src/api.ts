const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) || '/api';

export interface ResearcherProfile {
    n_id: number;
    name: string;
    email: string;
    department: string;
    school: string;
    one_line_summary?: string;
    topics?: string;
    research_area?: string;
    subfields?: string;
}

export interface Neighbour {
    n_id: number;
    name: string;
    department: string;
    school: string;
    similarity: number;
    rank: number;
}

export interface PublicationMatch {
    title: string;
    abstract?: string;
    similarity: number;
    doi?: string;
    main_url?: string;
    authors?: string;
}

export interface OwnPublication {
    title: string;
    abstract?: string;
    year?: number;
    doi?: string;
    main_url?: string;
    authors?: string;
}

export interface ExpertResearcherResult {
    n_id: number;
    name: string;
    school: string;
    department: string;
    similarity: number;
}

export interface ExpertPublicationResult {
    n_id: number;
    researcher_name?: string;
    school?: string;
    department?: string;
    publication_index?: number | null;
    article_id: string;
    title: string;
    abstract_snippet?: string;
    main_url?: string;
    similarity: number;
}

export interface ExpertSearchResponse {
    query: string;
    top_researchers: ExpertResearcherResult[];
    top_publications: ExpertPublicationResult[];
}

export interface ExpertThemeOption {
    option_id: string;
    label: string;
    scope: 'global' | 'school' | 'policy';
    school?: string | null;
    school_label?: string | null;
    topic_id: number;
}

export interface ExpertSearchFilters {
    schools: string[];
    departments_by_school: Record<string, string[]>;
}

export interface ReportTheme {
    theme_name: string;
    description: string;
}

export interface ReportOverviewV3 {
    core_identity: string;
    cross_cutting_themes: ReportTheme[];
    stats: {
        total_researchers: number;
        total_active_researchers: number;
        total_publications: number;
        total_recent_publications: number;
        num_schools: number;
        num_departments: number;
        avg_pubs_per_researcher: number;
        percent_recent: number;
        largest_school?: any;
        most_productive_school?: any;
    };
}

export interface ReportSchoolSummary {
    key: string;
    name: string;
    departments: number;
    researchers: number;
    active_researchers: number;
    publications: number;
    publications_recent?: number;
}

export interface ReportSchoolDetail {
    key: string;
    name: string;
    researchers: number;
    publications: number;
    publications_recent?: number;
    active_researchers: number;
    themes: ReportTheme[];
    policy_domains: string[];
    departments: {
        department: string;
        researchers: number;
        publications: number;
        avg_pubs: number;
        share_active: number;
    }[];
}

export interface ReportKeyArea {
    topic_id: number;
    theme_name: string;
    description: string;
    keywords: string[];
    count: number;
}

export interface ReportTopicExpert {
    n_id: number;
    name: string;
    school: string;
    department: string;
    similarity: number;
    assigned_topic?: number;
}

export interface ReportMacroTheme {
    theme_id: number;
    title: string;
    description: string;
    policy_relevance?: string;
    keywords: string[];
    funding_domains: string[];
    methodological_approaches: string[];
    schools_contributing: string[];
    n_researchers: number;
    stage1_cluster_ids: number[];
    aligned_research_areas: ReportKeyArea[];
}

export interface MapPointRecord {
    key: string;
    n_id: number;
    publication_index?: number | null;
    name?: string;
    school?: string;
    department?: string;
    topic?: number | null;
    topic_name?: string | null;
    policy_domain?: number | null;
    policy_name?: string | null;
    x: number;
    y: number;
    title?: string;
    abstract?: string;
    email?: string;
    one_line_summary?: string;
    research_area?: string;
    topics?: string;
    subfields?: string;
    centroid_similarity?: number | null;
}

export interface MapFiltersData {
    schools: string[];
    departments_by_school: Record<string, string[]>;
    researchers: { n_id: number; name: string; school?: string; department?: string }[];
    topics: { topic: number; topic_name?: string; count: number }[];
    policies: { policy_domain: number; policy_name?: string; count: number }[];
}

export type MapTopicsMap = Record<
    string,
    { topic_id: number; topic_name?: string; topic_description?: string; top_words?: string[] }
>;

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
    const response = await fetch(url, init);
    if (!response.ok) {
        throw new Error(`Request failed (${response.status}) for ${url}`);
    }
    return response.json() as Promise<T>;
}

export const fetchOverview = () => fetch(`${API_BASE}/report/overview`).then(r => r.json());
export const fetchSchools = () => fetch(`${API_BASE}/report/schools`).then(r => r.json());

export const searchResearchers = (q: string) => fetch(`${API_BASE}/network/search?q=${encodeURIComponent(q)}`).then(r => r.json());
export const lookupByEmail = (email: string) =>
    fetch(`${API_BASE}/network/lookup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
    }).then(r => r.json());

export const fetchResearcherProfile = (n_id: number) => fetch(`${API_BASE}/network/researcher/${n_id}`).then(r => r.json());

export const fetchNeighbours = (n_id: number, k: number = 10, excludeDept: boolean = false, excludeSchool: boolean = false) => {
    const params = new URLSearchParams({
        k: k.toString(),
        exclude_department: excludeDept.toString(),
        exclude_school: excludeSchool.toString()
    });
    return fetch(`${API_BASE}/network/neighbours/${n_id}?${params}`).then(r => r.json());
};

export const fetchMatchingPublications = (query_n_id: number, target_n_id: number) =>
    fetch(`${API_BASE}/network/publications/${query_n_id}/${target_n_id}`).then(r => r.json());

export const fetchMyPublications = (n_id: number) =>
    fetch(`${API_BASE}/network/my-publications/${n_id}`).then(r => r.json());

export const fetchMapResearchers = () => fetch(`${API_BASE}/map/researchers`).then(r => r.json());
export const fetchMapPublications = () => fetch(`${API_BASE}/map/publications`).then(r => r.json());
export const fetchMapV2Researchers = () => fetch(`${API_BASE}/map-v2/researchers`).then(r => r.json());
export const fetchMapPoints = () => fetchJson<MapPointRecord[]>(`${API_BASE}/map/points`);
export const fetchMapFilters = () => fetchJson<MapFiltersData>(`${API_BASE}/map/filters`);
export const fetchMapTopics = () => fetchJson<MapTopicsMap>(`${API_BASE}/map/topics`);
export const fetchMapV2Points = () => fetchJson<MapPointRecord[]>(`${API_BASE}/map-v2/points`);
export const fetchMapV2Filters = () => fetchJson<MapFiltersData>(`${API_BASE}/map-v2/filters`);
export const fetchMapV2Topics = () => fetchJson<MapTopicsMap>(`${API_BASE}/map-v2/topics`);

export const runExpertSearch = (
    query: string,
    topResearchers: number = 12,
    topPublications: number = 20,
    school?: string,
    department?: string
) =>
    fetchJson<ExpertSearchResponse>(`${API_BASE}/expert-search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            top_k_researchers: topResearchers,
            top_k_publications: topPublications,
            school: school || null,
            department: department || null,
        }),
    } as RequestInit);

export const fetchExpertThemeOptions = () => fetchJson<ExpertThemeOption[]>(`${API_BASE}/expert-search/themes/options`);
export const fetchExpertSearchFilters = () => fetchJson<ExpertSearchFilters>(`${API_BASE}/expert-search/filters`);

export const runExpertThemeSearch = (
    optionId: string,
    topResearchers: number = 12,
    topPublications: number = 20,
    school?: string,
    department?: string
) =>
    fetchJson<ExpertSearchResponse>(`${API_BASE}/expert-search/themes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            option_id: optionId,
            top_k_researchers: topResearchers,
            top_k_publications: topPublications,
            school: school || null,
            department: department || null,
        }),
    } as RequestInit);

export const fetchQueryRankedPublications = (n_id: number, query: string, limit: number = 15) =>
    fetchJson<ExpertPublicationResult[]>(
        `${API_BASE}/expert-search/researcher/${n_id}/publications?query=${encodeURIComponent(query)}&limit=${limit}`
    );

export const fetchReportV3Overview = () => fetchJson<ReportOverviewV3>(`${API_BASE}/report/v3/overview`);
export const fetchReportV3Statistics = () => fetchJson<any>(`${API_BASE}/report/v3/statistics`);
export const fetchReportV3Schools = () => fetchJson<ReportSchoolSummary[]>(`${API_BASE}/report/v3/schools`);
export const fetchReportV3School = (schoolKey: string) =>
    fetchJson<ReportSchoolDetail>(`${API_BASE}/report/v3/school/${encodeURIComponent(schoolKey)}`);
export const fetchReportV3SchoolTopics = (schoolKey: string) =>
    fetchJson<ReportKeyArea[]>(`${API_BASE}/report/v3/school/${encodeURIComponent(schoolKey)}/topics`);
export const fetchReportV3SchoolTopicExperts = (schoolKey: string, topicId: number, limit: number = 12) =>
    fetchJson<ReportTopicExpert[]>(
        `${API_BASE}/report/v3/school/${encodeURIComponent(schoolKey)}/topics/${topicId}/experts?limit=${limit}`
    );
export const fetchReportV3KeyAreas = () => fetchJson<ReportKeyArea[]>(`${API_BASE}/report/v3/key-areas`);
export const fetchReportV3KeyAreaExperts = (topicId: number, limit: number = 12) =>
    fetchJson<ReportTopicExpert[]>(`${API_BASE}/report/v3/key-areas/${topicId}/experts?limit=${limit}`);
export const fetchReportV3Themes = () => fetchJson<ReportMacroTheme[]>(`${API_BASE}/report/v3/themes`);
export const fetchReportV3ThemeExperts = (themeId: number, limit: number = 12) =>
    fetchJson<ReportTopicExpert[]>(`${API_BASE}/report/v3/themes/${themeId}/experts?limit=${limit}`);
export const reportV3PdfUrl = `${API_BASE}/report/v3/pdf`;
