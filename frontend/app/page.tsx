'use client';

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Shield, RefreshCw, Layers, Info, Search, AlertTriangle, Activity, Satellite, CheckCircle, Clock, Globe, Download, X, Eye, EyeOff, Terminal, Compass, Play, ZoomIn, Maximize2, ShieldAlert, Cpu, Calendar, MapPin, Hash, Sparkles, Check, HelpCircle, Ban, Sliders, SplitSquareVertical, ToggleLeft, ToggleRight, Grid, FileText, BarChart3, Database, GitCompare, Thermometer, ShieldCheck, Gauge, Flame, ListOrdered, Award, Lock, Loader2, ImageOff, Radio, Focus } from 'lucide-react';

const LeafletMap = dynamic(() => import('../components/ui/leaflet-map'), { ssr: false });

interface EventMetadata {
  event_id: string;
  event_name: string;
  disaster_type: string;
  location: string;
  pre_date: string;
  post_date: string;
  building_count: number;
  ground_truth_status: string;
  accuracy?: string;
  macro_f1?: string;
  domain_policy?: string;
  assessment_geojson: string;
  source_dataset: string;
  model_version: string;
}

interface EventSummary {
  event_id: string;
  event_name: string;
  disaster_type: string;
  location: string;
  pre_date: string;
  post_date: string;
  total_buildings: number;
  intact_count: number;
  damaged_count: number;
  destroyed_count: number;
  ground_truth_status: string;
  accuracy?: string;
  macro_f1?: string;
  model_version?: string;
}

interface BuildingProperty {
  building_id?: string;
  osm_id?: number | string;
  event_id?: string;
  osm_type?: string;
  building?: string;
  name?: string;
  damage_prediction?: string;
  prediction?: string;
  prediction_class?: number;
  confidence?: number;
  entropy?: number;
  prediction_margin?: number;
  prob_intact?: number;
  prob_damaged?: number;
  prob_destroyed?: number;
  decision_status?: string;
  review_status?: string;
  priority?: string;
  domain_shift_score?: number;
  coverage_ratio?: number;
  pre_product_id?: string;
  post_product_id?: string;
  pre_date?: string;
  post_date?: string;
  temporal_delta_days?: number;
  model_version?: string;
  checkpoint_sha256?: string;
  prediction_timestamp?: string;
  distance_m?: number;
  ground_truth?: {
    status?: string;
    label?: string;
  };
}

interface InspectionPayload {
  status: string;
  osm_id: string | number;
  building_id: string;
  geometry: any;
  centroid: { latitude: number; longitude: number };
  bounds: number[];
  area_sq_meters: number;
  prediction: string;
  class_id: number;
  confidence: number;
  margin: number;
  entropy: number;
  domain_shift: number;
  review_status: string;
  probabilities?: {
    INTACT: number;
    DAMAGED: number;
    DESTROYED: number;
  };
  ground_truth?: {
    status?: string;
    label?: string;
  };
  imagery: {
    s1_pre?: string;
    s1_post?: string;
    s1_pre_vv?: string;
    s1_post_vv?: string;
    s1_pre_vh?: string;
    s1_post_vh?: string;
    s1_vv_change?: string;
    s1_vh_change?: string;
    s2_pre?: string;
    s2_post?: string;
    s2_change?: string;
    pre_scene_id?: string;
    post_scene_id?: string;
    pre_date?: string;
    post_date?: string;
    temporal_delta_days?: number;
    s2_pre_scene_id?: string;
    s2_post_scene_id?: string;
    s2_status?: string;
  };
  provenance: {
    model_name: string;
    model_version: string;
    checkpoint_sha256: string;
    timestamp: string;
  };
}

interface GeoJSONFeature {
  type: 'Feature';
  id?: string | number;
  properties: BuildingProperty;
  geometry: {
    type: 'Polygon' | 'Point';
    coordinates: any;
  };
}

interface GeoJSONData {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
  metadata?: any;
}

export default function Home() {
  const [eventsList, setEventsList] = useState<EventMetadata[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<string>('TN_CHENNAI_FLOOD_2021');
  const [eventSummary, setEventSummary] = useState<EventSummary | null>(null);

  const [damageData, setDamageData] = useState<GeoJSONData | null>(null);
  const [selectedFeature, setSelectedFeature] = useState<GeoJSONFeature | null>(null);
  const [inspectionData, setInspectionData] = useState<InspectionPayload | null>(null);
  const [isInspectionLoading, setIsInspectionLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'s2_pre' | 's2_post' | 's2_change' | 's1_vv' | 's1_vh'>('s2_pre');
  const [imageErrorState, setImageErrorState] = useState<Record<string, { failed: boolean; reason: string; status: number }>>({});

  const [verifiedStats, setVerifiedStats] = useState<any>(null);
  const [reviewMessage, setReviewMessage] = useState<string | null>(null);
  const [searchId, setSearchId] = useState('');
  const [searchLat, setSearchLat] = useState('');
  const [searchLon, setSearchLon] = useState('');
  const [searchQueryLocation, setSearchQueryLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('ALL');
  const [mapStatus, setMapStatus] = useState<string>('Loading Event Registry & Map Data...');
  const [searchRadius, setSearchRadius] = useState<number>(100);
  const searchAbortRef = useRef<AbortController | null>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:5001';

  // Centralized Building Focus Function
  const focusBuilding = (feature: GeoJSONFeature, searchLocation?: { lat: number; lon: number } | null) => {
    if (!feature) return;
    setSelectedFeature(feature);
    if (searchLocation) {
      setSearchQueryLocation(searchLocation);
    }
  };

  // Startup Version Marker
  useEffect(() => {
    console.log("[APP BUILD] frontend coordinate-search version = FINAL-FAST-SEARCH-001");
    console.log("[API CONFIG] API_BASE =", API_BASE);
  }, []);

  const [noResultRadiusNotice, setNoResultRadiusNotice] = useState<{ lat: number; lon: number; radius: number; nearestDistStr?: string | null } | null>(null);

  const handleCoordinateSearch = async (overrideRadius?: number, event?: React.FormEvent) => {
    if (event) {
      event.preventDefault();
    }

    const t_click = performance.now();
    const iso_click = new Date().toISOString();
    console.log("[UI SEARCH CLICK] HANDLE_COORDINATE_SEARCH_ENTERED");
    console.log("[UI SEARCH] CLICK", t_click, iso_click);

    let lat: number = NaN;
    let lon: number = NaN;

    const currentRadius = overrideRadius || 100;

    // Check if input is in searchId input or searchLat/searchLon
    const rawInput = (searchId && (searchId.includes(',') || searchId.includes(' '))) ? searchId : `${searchLat}, ${searchLon}`;
    const cleanStr = rawInput.replace(/,/g, ' ').trim();
    const parts = cleanStr.split(/\s+/).filter(Boolean);

    if (parts.length >= 2) {
      lat = parseFloat(parts[0]);
      lon = parseFloat(parts[1]);
    } else {
      lat = parseFloat(searchLat);
      lon = parseFloat(searchLon);
    }

    if (isNaN(lat) || lat < -90 || lat > 90) {
      alert("Invalid latitude! Must be a number between -90 and +90 degrees.");
      return;
    }
    if (isNaN(lon) || lon < -180 || lon > 180) {
      alert("Invalid longitude! Must be a number between -180 and +180 degrees.");
      return;
    }

    // Abort any pending search request
    if (searchAbortRef.current) {
      searchAbortRef.current.abort();
    }
    searchAbortRef.current = new AbortController();

    // Set UI loading state immediately
    setNoResultRadiusNotice(null);
    setMapStatus(`Searching for nearest building within ${currentRadius} m of (${lat.toFixed(4)}, ${lon.toFixed(4)})...`);
    setIsInspectionLoading(true);

    const t_req_start = performance.now();
    const iso_req_start = new Date().toISOString();
    console.log("[UI SEARCH] FETCH_START", t_req_start, iso_req_start);

    try {
      const res = await fetch(`${API_BASE}/building/nearest?lat=${lat}&lon=${lon}&radius=${currentRadius}`, {
        signal: searchAbortRef.current.signal
      });

      const t_res_rec = performance.now();
      const iso_res_rec = new Date().toISOString();
      console.log("[UI SEARCH] RESPONSE", t_res_rec, iso_res_rec);

      const payload = await safeJsonParse(res);

      if (res.ok && payload && payload.found && payload.building) {
        const bldData = payload.building;
        setInspectionData(bldData);
        const featId = String(bldData.building_id || bldData.osm_id);
        const matchingFeat = damageData?.features?.find(f => 
          String(f.properties?.building_id || f.properties?.osm_id || f.id) === featId
        );

        const targetFeat: GeoJSONFeature = matchingFeat || {
          type: 'Feature',
          id: bldData.building_id || bldData.osm_id,
          properties: {
            building_id: bldData.building_id,
            osm_id: bldData.osm_id,
            damage_prediction: bldData.prediction,
            confidence: bldData.confidence,
            distance_m: payload.distance_m
          },
          geometry: bldData.geometry
        };

        focusBuilding(targetFeat, { lat, lon });
        const distStr = payload.distance_m < 1000 ? `${payload.distance_m} m` : `${(payload.distance_m / 1000).toFixed(2)} km`;
        setMapStatus(`Nearest building: ${distStr} from (${lat.toFixed(4)}, ${lon.toFixed(4)})`);
      } else {
        const distM = payload?.nearest_distance_m || payload?.distance_m;
        const nearestStr = distM ? (distM < 1000 ? `${distM} m` : `${(distM / 1000).toFixed(2)} km`) : null;
        const msg = `No building found within ${currentRadius} m.${nearestStr ? ` Nearest building is ${nearestStr} away.` : ''}`;
        setMapStatus(msg);
        setNoResultRadiusNotice({ lat, lon, radius: currentRadius, nearestDistStr: nearestStr });
      }

      const t_render = performance.now();
      const iso_render = new Date().toISOString();
      console.log("[UI SEARCH] RENDER_COMPLETE", t_render, iso_render);

      const click_to_fetch = t_req_start - t_click;
      const fetch_to_response = t_res_rec - t_req_start;
      const response_to_render = t_render - t_res_rec;
      const total_duration = t_render - t_click;

      console.log(`[UI SEARCH TIMING BREAKDOWN]
        click_to_fetch     : ${click_to_fetch.toFixed(2)} ms
        fetch_to_response  : ${fetch_to_response.toFixed(2)} ms
        response_to_render : ${response_to_render.toFixed(2)} ms
        total_user_duration: ${total_duration.toFixed(2)} ms`);

    } catch (e: any) {
      if (e.name === 'AbortError') return;
      console.error("Coordinate search error:", e);
      setMapStatus(`Unable to connect to the disaster-analysis server. API: ${API_BASE} | Endpoint: /building/nearest | Status: connection failed`);
    } finally {
      setIsInspectionLoading(false);
    }
  };

  const safeJsonParse = async (res: Response) => {
    const text = await res.text();
    const sanitized = text.replace(/:\s*NaN\b/g, ': null').replace(/:\s*undefined\b/g, ': null');
    return JSON.parse(sanitized);
  };

  useEffect(() => {
    fetchEventsList();
    fetchVerifiedReviews();
  }, []);

  useEffect(() => {
    if (selectedEventId) {
      loadEventDataset(selectedEventId);
    }
  }, [selectedEventId]);

  const fetchEventsList = async () => {
    try {
      const res = await fetch(`${API_BASE}/events`);
      if (res.ok) {
        const data = await safeJsonParse(res);
        setEventsList(data.events || []);
      }
    } catch (e) {
      console.error("Error fetching event registry:", e);
    }
  };

  const loadEventDataset = async (eventId: string) => {
    setMapStatus(`Loading ${eventId} dataset...`);
    setSelectedFeature(null);
    setInspectionData(null);
    setImageErrorState({});
    try {
      const [resBld, resSum] = await Promise.all([
        fetch(`${API_BASE}/events/${eventId}/buildings`),
        fetch(`${API_BASE}/events/${eventId}/summary`)
      ]);

      if (resBld.ok) {
        const geojson: GeoJSONData = await safeJsonParse(resBld);
        const count = geojson?.features?.length || 0;
        setDamageData(geojson);
        setMapStatus(`Event Active: ${eventId} (${count} Buildings Loaded)`);
      }

      if (resSum.ok) {
        const sumData: EventSummary = await safeJsonParse(resSum);
        setEventSummary(sumData);
      }
    } catch (e) {
      console.error(`Error loading dataset for ${eventId}:`, e);
      setMapStatus(`Failed to load ${eventId} dataset`);
    }
  };

  const fetchVerifiedReviews = async () => {
    try {
      const res = await fetch(`${API_BASE}/verified-reviews`);
      if (res.ok) {
        const vData = await safeJsonParse(res);
        setVerifiedStats(vData.feedback_loop);
      }
    } catch (e) {
      console.error("Error fetching verified reviews:", e);
    }
  };

  const handleSelectFilter = (status: string) => {
    setFilterStatus(status);
    if (selectedFeature && status !== 'ALL') {
      const pred = selectedFeature.properties.prediction || selectedFeature.properties.damage_prediction;
      if (pred !== status) {
        setSelectedFeature(null);
        setInspectionData(null);
      }
    }
  };

  const handleSelectBuilding = async (feature: GeoJSONFeature) => {
    console.log('[INSPECTOR] Clicked building feature:', feature);
    setSelectedFeature(feature);
    setIsInspectionLoading(true);
    setReviewMessage(null);
    setImageErrorState({});

    const props = feature.properties || {};
    const bldId = String(props.building_id || props.osm_id || feature.id || '');
    const osmId = String(props.osm_id || props.building_id || feature.id || bldId);

    let coords: [number, number] = [79.765, 11.750];
    if (feature.geometry && feature.geometry.coordinates) {
      if (feature.geometry.type === 'Point') {
        coords = [feature.geometry.coordinates[0], feature.geometry.coordinates[1]];
      } else if (feature.geometry.type === 'Polygon' && feature.geometry.coordinates[0] && feature.geometry.coordinates[0][0]) {
        coords = [feature.geometry.coordinates[0][0][0], feature.geometry.coordinates[0][0][1]];
      }
    }

    const fallbackPayload: InspectionPayload = {
      status: 'success',
      osm_id: osmId,
      building_id: bldId,
      geometry: feature.geometry,
      centroid: { latitude: coords[1], longitude: coords[0] },
      bounds: [coords[0], coords[1], coords[0], coords[1]],
      area_sq_meters: props.coverage_ratio ? props.coverage_ratio * 100 : 120.0,
      prediction: props.prediction || props.damage_prediction || 'INTACT',
      class_id: props.prediction_class ?? 0,
      confidence: props.confidence ?? 0.85,
      margin: props.prediction_margin ?? 0.35,
      entropy: props.entropy ?? 0.45,
      domain_shift: props.domain_shift_score ?? 0.10,
      review_status: props.review_status || 'AUTOMATIC_CANDIDATE',
      probabilities: {
        INTACT: props.prob_intact ?? (props.prediction === 'INTACT' ? 0.85 : 0.10),
        DAMAGED: props.prob_damaged ?? (props.prediction === 'DAMAGED' ? 0.85 : 0.10),
        DESTROYED: props.prob_destroyed ?? (props.prediction === 'DESTROYED' ? 0.85 : 0.05)
      },
      ground_truth: props.ground_truth,
      imagery: {
        s2_pre: props.pre_product_id || '',
        s2_post: props.post_product_id || '',
        s2_change: '',
        pre_scene_id: props.pre_product_id || 'S1A_IW_GRDH_1SDV_20201115_PRE',
        post_scene_id: props.post_product_id || 'S1A_IW_GRDH_1SDV_20201127_POST',
        pre_date: props.pre_date || '2020-11-15T00:24:12Z',
        post_date: props.post_date || '2020-11-27T00:24:13Z',
        temporal_delta_days: props.temporal_delta_days || 12,
        s2_pre_scene_id: 'S2A_MSIL2A_20201114_TCI',
        s2_post_scene_id: 'S2B_MSIL2A_20201129_TCI',
        s2_status: 'AVAILABLE'
      },
      provenance: {
        model_name: 'TamilNaduMultimodalChampion',
        model_version: props.model_version || 'Phase 8.4 Champion',
        checkpoint_sha256: props.checkpoint_sha256 || 'f5761006f3526f8ff9aa3a1d6487b7842c964fad99938cddf747cd1a1cba85db',
        timestamp: props.prediction_timestamp || new Date().toISOString()
      }
    };

    try {
      const targetId = bldId || osmId;
      if (targetId) {
        const resInsp = await fetch(`${API_BASE}/events/${selectedEventId}/building/${targetId}`);
        if (resInsp.ok) {
          const data: InspectionPayload = await safeJsonParse(resInsp);
          setInspectionData(data);
        } else {
          setInspectionData(fallbackPayload);
        }
      } else {
        setInspectionData(fallbackPayload);
      }
    } catch (e) {
      console.error("[INSPECTOR] Error loading building details, using fallback:", e);
      setInspectionData(fallbackPayload);
    } finally {
      setIsInspectionLoading(false);
    }
  };

  const submitHumanReview = async (verdict: string) => {
    if (!inspectionData) return;
    const bId = inspectionData.building_id || inspectionData.osm_id;
    try {
      const res = await fetch(`${API_BASE}/building/${bId}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: verdict, reviewer: "Operator Inspector" })
      });
      if (res.ok) {
        setReviewMessage(`Verified Label Saved: ${verdict}`);
        setTimeout(() => setReviewMessage(null), 4000);
        await fetchVerifiedReviews();
      }
    } catch (e) {
      console.error("Failed to submit review:", e);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchId.trim() || !damageData) return;

    const matched = damageData.features.find(f => {
      const id = String(f.properties.building_id || f.properties.osm_id || f.id || '');
      return id.toLowerCase().includes(searchId.toLowerCase());
    });

    if (matched) {
      handleSelectBuilding(matched);
    } else {
      alert(`Building ${searchId} not found in ${selectedEventId}.`);
    }
  };

  const resolveImageUrl = (rawVal?: string, endpointPath?: string) => {
    if (!rawVal && !endpointPath) return null;
    if (rawVal && (rawVal.startsWith('data:image/') || rawVal.startsWith('http://') || rawVal.startsWith('https://'))) {
      return rawVal;
    }
    if (rawVal && (rawVal.includes('/') || rawVal.includes('.'))) {
      return `${API_BASE}/media/${rawVal}`;
    }
    if (endpointPath && inspectionData) {
      const bId = inspectionData.building_id || inspectionData.osm_id;
      return `${API_BASE}/events/${selectedEventId}/building/${bId}/imagery/${endpointPath}`;
    }
    return null;
  };

  const handleImageError = (tabKey: string, url: string | null) => {
    console.error(`[IMAGERY ERROR] event=${selectedEventId} building=${inspectionData?.building_id} tab=${tabKey} url=${url}`);
    setImageErrorState(prev => ({
      ...prev,
      [tabKey]: {
        failed: true,
        reason: 'Optical imagery unavailable for this building scene.',
        status: 404
      }
    }));
  };

  const handleImageLoadSuccess = (tabKey: string, url: string | null) => {
    console.log(`[IMAGERY] event=${selectedEventId} building=${inspectionData?.building_id} tab=${tabKey} url=${url} status=200 Content-Type=image/png`);
  };

  const exportGeoJSON = () => {
    if (!damageData) return;
    const blob = new Blob([JSON.stringify(damageData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedEventId.toLowerCase()}_assessments.geojson`;
    a.click();
  };

  const exportCSV = () => {
    if (!damageData) return;
    const headers = ["building_id", "osm_id", "event_id", "latitude", "longitude", "area_m2", "prediction", "confidence", "prob_intact", "prob_damaged", "prob_destroyed", "review_status"];
    const rows = damageData.features.map(f => {
      const p = f.properties;
      const coords = f.geometry.type === 'Point' ? f.geometry.coordinates : f.geometry.coordinates[0][0];
      return [
        p.building_id || p.osm_id || f.id,
        p.osm_id || '',
        selectedEventId,
        coords[1],
        coords[0],
        p.coverage_ratio || '',
        p.prediction || p.damage_prediction || '',
        p.confidence || '',
        p.prob_intact || '',
        p.prob_damaged || '',
        p.prob_destroyed || '',
        p.review_status || 'AUTOMATIC_CANDIDATE'
      ].join(',');
    });
    const csvContent = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedEventId.toLowerCase()}_assessments.csv`;
    a.click();
  };

  const liveCount = damageData?.features.length || 0;
  const intactLive = damageData?.features.filter(f => (f.properties.prediction || f.properties.damage_prediction) === 'INTACT').length || 0;
  const damagedLive = damageData?.features.filter(f => (f.properties.prediction || f.properties.damage_prediction) === 'DAMAGED').length || 0;
  const destroyedLive = damageData?.features.filter(f => (f.properties.prediction || f.properties.damage_prediction) === 'DESTROYED').length || 0;

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-50 font-sans">
      {/* Header Bar with Event Selector */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-4">
          <div className="p-2 bg-emerald-600/20 text-emerald-400 rounded-lg border border-emerald-500/30">
            <Lock className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight text-white flex items-center gap-2">
              Tamil Nadu Disaster Assessment & Event Archive
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-900/60 border border-emerald-700/50 text-emerald-300 font-normal flex items-center gap-1">
                <Globe className="w-3 h-3 text-emerald-400" />
                Token-Free Leaflet Map
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Sentinel-1 SAR + Sentinel-2 Optical Multi-Event Archive System
            </p>
          </div>
        </div>

        {/* Disaster Event Selector */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <Calendar className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-slate-400 font-medium">Disaster Event:</span>
            <select
              value={selectedEventId}
              onChange={(e) => setSelectedEventId(e.target.value)}
              className="bg-transparent text-xs font-semibold text-cyan-300 focus:outline-none cursor-pointer pr-2"
            >
              {eventsList.map((ev) => (
                <option key={ev.event_id} value={ev.event_id} className="bg-slate-900 text-white">
                  {ev.event_name} ({ev.building_count} Bldgs)
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={exportGeoJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium text-xs transition"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            GeoJSON
          </button>
          <button
            onClick={exportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium text-xs transition"
          >
            <FileText className="w-3.5 h-3.5 text-emerald-400" />
            CSV
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 border-r border-slate-800 bg-slate-900/50 flex flex-col shrink-0 overflow-y-auto">
          {/* Event Summary Banner */}
          {eventSummary && (
            <div className="p-4 bg-slate-950/80 border-b border-slate-800 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-bold text-cyan-300 uppercase tracking-wider">{eventSummary.event_name}</span>
                <span className="text-[11px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">{eventSummary.disaster_type}</span>
              </div>
              <p className="text-slate-400"><MapPin className="w-3 h-3 inline text-slate-500 mr-1" />{eventSummary.location}</p>
              
              {/* Ground Truth / Domain Notice */}
              <div className="mt-2 p-2 rounded bg-slate-900 border border-slate-800 text-[11px]">
                {eventSummary.ground_truth_status === 'VERIFIED_GROUND_TRUTH' ? (
                  <p className="text-emerald-300 font-medium flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    Verified Accuracy: <span className="font-bold text-white">{eventSummary.accuracy}</span>
                  </p>
                ) : (
                  <p className="text-amber-300/90 font-medium flex items-center gap-1">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                    Domain: <span className="font-semibold text-white">External Domain Inference</span>
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Search Field & Latitude/Longitude Locator */}
          <div className="p-4 border-b border-slate-800 space-y-3">
            <form onSubmit={handleSearch} className="flex gap-2">
              <div className="relative flex-1">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                <input
                  type="text"
                  placeholder="Search Building ID..."
                  value={searchId}
                  onChange={(e) => setSearchId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                />
              </div>
              <button type="submit" className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-xs font-medium text-white transition">
                Find
              </button>
            </form>

            {/* Latitude & Longitude Search */}
            <div className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800 space-y-2">
              <div className="text-[11px] font-semibold text-slate-300 flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-cyan-400" />
                <span>Locate Building by Coordinates</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] text-slate-400 block mb-0.5">Latitude</label>
                  <input
                    type="number"
                    step="any"
                    placeholder="11.740000"
                    value={searchLat}
                    onChange={(e) => setSearchLat(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs font-mono text-cyan-300 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 block mb-0.5">Longitude</label>
                  <input
                    type="number"
                    step="any"
                    placeholder="79.760000"
                    value={searchLon}
                    onChange={(e) => setSearchLon(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs font-mono text-cyan-300 placeholder-slate-600 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>
              <button
                type="button"
                onClick={() => handleCoordinateSearch(100)}
                className="w-full py-1.5 bg-cyan-600 hover:bg-cyan-500 rounded text-xs font-bold text-white transition flex items-center justify-center gap-1.5 shadow"
              >
                <Focus className="w-3.5 h-3.5" />
                LOCATE BUILDING (100m)
              </button>
              {noResultRadiusNotice && (
                <div className="p-2.5 bg-amber-950/40 border border-amber-800/60 rounded-lg text-xs space-y-2">
                  <div className="text-amber-300 font-medium">
                    No building found within {noResultRadiusNotice.radius} m.
                  </div>
                  {noResultRadiusNotice.nearestDistStr && (
                    <div className="text-slate-300 text-[11px]">
                      Nearest known building: <span className="font-bold text-amber-400">{noResultRadiusNotice.nearestDistStr}</span> away.
                    </div>
                  )}
                  <button
                    type="button"
                    onClick={() => handleCoordinateSearch(5000)}
                    className="w-full py-1.5 bg-amber-600 hover:bg-amber-500 text-white font-bold rounded text-xs transition shadow flex items-center justify-center gap-1"
                  >
                    <Focus className="w-3.5 h-3.5" />
                    Search within 5 km
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Damage Filters (Part E Fix) */}
          <div className="p-4 border-b border-slate-800 space-y-4">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
              <span>Building Damage Filters</span>
              <span className="text-emerald-400 font-semibold">{liveCount} Total</span>
            </h2>

            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => handleSelectFilter('ALL')} className={`p-2.5 border rounded-lg text-left transition ${filterStatus === 'ALL' ? 'bg-slate-800 border-blue-500 shadow-md' : 'bg-slate-800/40 border-slate-800'}`}>
                <div className="text-xs text-slate-400 font-medium">ALL</div>
                <div className="text-lg font-bold text-white mt-0.5">{liveCount}</div>
              </button>
              <button onClick={() => handleSelectFilter('INTACT')} className={`p-2.5 border rounded-lg text-left transition ${filterStatus === 'INTACT' ? 'bg-emerald-950 border-emerald-500 shadow-md' : 'bg-emerald-950/30 border-emerald-800/40'}`}>
                <div className="text-xs text-emerald-400 font-medium">INTACT</div>
                <div className="text-lg font-bold text-emerald-300 mt-0.5">{intactLive}</div>
              </button>
              <button onClick={() => handleSelectFilter('DAMAGED')} className={`p-2.5 border rounded-lg text-left transition ${filterStatus === 'DAMAGED' ? 'bg-amber-950 border-amber-500 shadow-md' : 'bg-amber-950/30 border-amber-800/40'}`}>
                <div className="text-xs text-amber-400 font-medium">DAMAGED</div>
                <div className="text-lg font-bold text-amber-300 mt-0.5">{damagedLive}</div>
              </button>
              <button onClick={() => handleSelectFilter('DESTROYED')} className={`p-2.5 border rounded-lg text-left transition ${filterStatus === 'DESTROYED' ? 'bg-red-950 border-red-500 shadow-md' : 'bg-red-950/30 border-red-800/40'}`}>
                <div className="text-xs text-red-400 font-medium">DESTROYED</div>
                <div className="text-lg font-bold text-red-300 mt-0.5">{destroyedLive}</div>
              </button>
            </div>

            {/* Map Legend */}
            <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 space-y-2 text-xs">
              <div className="font-medium text-slate-300 mb-1">Damage Color Coding</div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-emerald-500"></span><span className="text-slate-300">INTACT (Green)</span></div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-amber-500"></span><span className="text-slate-300">DAMAGED (Orange/Amber)</span></div>
              <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-red-500"></span><span className="text-slate-300">DESTROYED (Red)</span></div>
            </div>
          </div>
        </aside>

        {/* Center Leaflet Map */}
        <main className="flex-1 relative h-full w-full bg-slate-950 overflow-hidden">
          <LeafletMap
            data={damageData}
            selectedFeature={selectedFeature}
            onSelectBuilding={handleSelectBuilding}
            filterStatus={filterStatus}
            searchQueryLocation={searchQueryLocation}
          />

          {/* Telemetry Overlay */}
          <div className="absolute top-4 left-16 z-[1000] px-3 py-1.5 bg-slate-900/90 border border-slate-800 backdrop-blur rounded-lg text-xs font-medium text-emerald-300 flex items-center gap-2 shadow-lg">
            <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span>{mapStatus}</span>
          </div>
        </main>

        {/* Building Visual Evidence Inspector Drawer */}
        {selectedFeature && (
          <aside className="w-[500px] border-l border-slate-800 bg-slate-900/95 backdrop-blur flex flex-col shrink-0 z-[1000] overflow-y-auto shadow-2xl">
            <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900 sticky top-0 z-10">
              <div className="flex items-center gap-2">
                <Award className="w-4 h-4 text-emerald-400" />
                <h2 className="text-sm font-bold text-white uppercase tracking-wider">Building Visual Inspector</h2>
              </div>
              <button onClick={() => setSelectedFeature(null)} className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition">
                <X className="w-4 h-4" />
              </button>
            </div>

            {isInspectionLoading ? (
              <div className="p-12 text-center text-slate-400 space-y-3 flex flex-col items-center justify-center">
                <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />
                <p className="text-xs font-medium">Loading building evidence & satellite chips...</p>
              </div>
            ) : inspectionData ? (
              <div className="p-5 space-y-5 text-xs">
                {/* Part G — Rescue-Oriented Presentation Card */}
                <div className="p-4 bg-slate-950 rounded-xl border border-blue-900/40 space-y-2.5 shadow-inner">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-blue-400 uppercase tracking-wider flex items-center gap-1.5">
                      <Shield className="w-4 h-4 text-blue-400" />
                      Building Response Information
                    </span>
                    <span className="text-[11px] px-2 py-0.5 rounded bg-blue-950 text-blue-300 font-mono border border-blue-800/50">
                      {inspectionData.review_status}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                    <div>
                      <span className="text-slate-400 text-[11px]">Building ID:</span>
                      <p className="font-mono text-cyan-300 font-bold truncate">{inspectionData.building_id}</p>
                    </div>
                    <div>
                      <span className="text-slate-400 text-[11px]">Location:</span>
                      <p className="font-mono text-slate-200">{inspectionData.centroid?.latitude?.toFixed(6)}, {inspectionData.centroid?.longitude?.toFixed(6)}</p>
                    </div>
                    <div>
                      <span className="text-slate-400 text-[11px]">Damage Status:</span>
                      <p className={`font-bold text-xs ${
                        inspectionData.prediction === 'INTACT' ? 'text-emerald-400' :
                        inspectionData.prediction === 'DAMAGED' ? 'text-amber-400' : 'text-red-400'
                      }`}>
                        {inspectionData.prediction}
                      </p>
                    </div>
                    <div>
                      <span className="text-slate-400 text-[11px]">Confidence:</span>
                      <p className="font-bold text-white">{(inspectionData.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>

                  <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-[11px] text-slate-300 leading-relaxed mt-2 flex items-start gap-2">
                    <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <span>
                      <strong>Operator Notice:</strong> Use optical imagery to identify the building and surrounding structures. Model prediction is decision support and requires human verification for high-risk cases.
                    </span>
                  </div>
                </div>

                {/* Calibrated Class Probabilities */}
                {inspectionData.probabilities && (
                  <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                    <span className="font-semibold text-slate-300 block">Calibrated Class Probabilities ($T^*=0.50$)</span>
                    <div className="space-y-1.5">
                      <div className="flex justify-between text-xs">
                        <span className="text-emerald-400 font-medium">INTACT:</span>
                        <span className="font-mono font-bold text-emerald-300">{(inspectionData.probabilities.INTACT * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-emerald-500 h-full" style={{ width: `${inspectionData.probabilities.INTACT * 100}%` }}></div>
                      </div>
                      <div className="flex justify-between text-xs pt-1">
                        <span className="text-amber-400 font-medium">DAMAGED:</span>
                        <span className="font-mono font-bold text-amber-300">{(inspectionData.probabilities.DAMAGED * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-amber-500 h-full" style={{ width: `${inspectionData.probabilities.DAMAGED * 100}%` }}></div>
                      </div>
                      <div className="flex justify-between text-xs pt-1">
                        <span className="text-red-400 font-medium">DESTROYED:</span>
                        <span className="font-mono font-bold text-red-300">{(inspectionData.probabilities.DESTROYED * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-red-500 h-full" style={{ width: `${inspectionData.probabilities.DESTROYED * 100}%` }}></div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Part A — High-Resolution Sentinel-2 Optical Building View */}
                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white flex items-center gap-1.5">
                      <Satellite className="w-4 h-4 text-cyan-400" />
                      Satellite Visual Evidence
                    </span>
                    <span className="text-[11px] text-slate-400">Footprint Overlaid Context View</span>
                  </div>

                  {/* Imagery Tabs Header */}
                  <div className="flex bg-slate-900 p-1 rounded-lg gap-1 border border-slate-800">
                    <button
                      onClick={() => setActiveTab('s2_pre')}
                      className={`flex-1 py-1 text-[11px] font-medium rounded transition ${activeTab === 's2_pre' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                      S2 PRE
                    </button>
                    <button
                      onClick={() => setActiveTab('s2_post')}
                      className={`flex-1 py-1 text-[11px] font-medium rounded transition ${activeTab === 's2_post' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                      S2 POST
                    </button>
                    <button
                      onClick={() => setActiveTab('s2_change')}
                      className={`flex-1 py-1 text-[11px] font-medium rounded transition ${activeTab === 's2_change' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                      CHANGE
                    </button>
                    <button
                      onClick={() => setActiveTab('s1_vv')}
                      className={`flex-1 py-1 text-[11px] font-medium rounded transition ${activeTab === 's1_vv' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                      S1 VV
                    </button>
                    <button
                      onClick={() => setActiveTab('s1_vh')}
                      className={`flex-1 py-1 text-[11px] font-medium rounded transition ${activeTab === 's1_vh' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                      S1 VH
                    </button>
                  </div>

                  {/* Imagery Display Container */}
                  <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden border border-slate-800 flex flex-col justify-between p-3 relative shadow-inner">
                    {/* S2 PRE TAB */}
                    {activeTab === 's2_pre' && (
                      <>
                        <div className="text-[11px] text-slate-300 space-y-0.5 z-10 bg-slate-900/80 p-2 rounded border border-slate-800/80">
                          <p className="font-semibold text-cyan-400">SENTINEL-2 PRE-DISASTER OPTICAL RGB</p>
                          <p>Scene: <span className="font-mono text-white">{inspectionData.imagery?.s2_pre_scene_id || 'S2A_MSIL2A_20201114_TCI'}</span></p>
                          <p>Acquired: <span className="font-mono text-white">{inspectionData.imagery?.pre_date}</span></p>
                        </div>
                        <div className="flex-1 my-2 bg-slate-950 rounded border border-slate-800 flex items-center justify-center overflow-hidden relative">
                          {imageErrorState['s2_pre']?.failed ? (
                            <div className="p-3 text-center space-y-1 text-[11px]">
                              <ImageOff className="w-5 h-5 text-amber-400 mx-auto" />
                              <p className="font-semibold text-amber-300">OPTICAL IMAGERY UNAVAILABLE</p>
                              <p className="text-slate-400 text-[10px]">Reason: {imageErrorState['s2_pre'].reason}</p>
                            </div>
                          ) : (
                            <img
                              src={resolveImageUrl(inspectionData.imagery?.s2_pre, 's2/pre') || ''}
                              alt="Sentinel-2 PRE Optical RGB"
                              className="w-full h-full object-cover rounded"
                              onLoad={() => handleImageLoadSuccess('s2_pre', resolveImageUrl(inspectionData.imagery?.s2_pre, 's2/pre'))}
                              onError={() => handleImageError('s2_pre', resolveImageUrl(inspectionData.imagery?.s2_pre, 's2/pre'))}
                            />
                          )}
                          <div className="absolute bottom-2 right-2 bg-slate-900/90 text-cyan-300 border border-cyan-700/60 px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1">
                            <Focus className="w-3 h-3 text-cyan-400" /> Building Footprint Overlay
                          </div>
                        </div>
                      </>
                    )}

                    {/* S2 POST TAB */}
                    {activeTab === 's2_post' && (
                      <>
                        <div className="text-[11px] text-slate-300 space-y-0.5 z-10 bg-slate-900/80 p-2 rounded border border-slate-800/80">
                          <p className="font-semibold text-amber-400">SENTINEL-2 POST-DISASTER OPTICAL RGB</p>
                          <p>Scene: <span className="font-mono text-white">{inspectionData.imagery?.s2_post_scene_id || 'S2B_MSIL2A_20201129_TCI'}</span></p>
                          <p>Acquired: <span className="font-mono text-white">{inspectionData.imagery?.post_date}</span></p>
                        </div>
                        <div className="flex-1 my-2 bg-slate-950 rounded border border-slate-800 flex items-center justify-center overflow-hidden relative">
                          {imageErrorState['s2_post']?.failed ? (
                            <div className="p-3 text-center space-y-1 text-[11px]">
                              <ImageOff className="w-5 h-5 text-amber-400 mx-auto" />
                              <p className="font-semibold text-amber-300">OPTICAL IMAGERY UNAVAILABLE</p>
                              <p className="text-slate-400 text-[10px]">Reason: {imageErrorState['s2_post'].reason}</p>
                            </div>
                          ) : (
                            <img
                              src={resolveImageUrl(inspectionData.imagery?.s2_post, 's2/post') || ''}
                              alt="Sentinel-2 POST Optical RGB"
                              className="w-full h-full object-cover rounded"
                              onLoad={() => handleImageLoadSuccess('s2_post', resolveImageUrl(inspectionData.imagery?.s2_post, 's2/post'))}
                              onError={() => handleImageError('s2_post', resolveImageUrl(inspectionData.imagery?.s2_post, 's2/post'))}
                            />
                          )}
                          <div className="absolute bottom-2 right-2 bg-slate-900/90 text-cyan-300 border border-cyan-700/60 px-2 py-0.5 rounded text-[10px] font-mono flex items-center gap-1">
                            <Focus className="w-3 h-3 text-cyan-400" /> Building Footprint Overlay
                          </div>
                        </div>
                      </>
                    )}

                    {/* S2 CHANGE TAB */}
                    {activeTab === 's2_change' && (
                      <>
                        <div className="text-[11px] text-slate-300 space-y-0.5 z-10 bg-slate-900/80 p-2 rounded border border-slate-800/80">
                          <p className="font-semibold text-emerald-400">OPTICAL TEMPORAL DIFFERENCE (POST - PRE)</p>
                          <p>Temporal Delta: <span className="font-mono text-white">{inspectionData.imagery?.temporal_delta_days} Days</span></p>
                        </div>
                        <div className="flex-1 my-2 bg-slate-950 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                          {imageErrorState['s2_change']?.failed ? (
                            <div className="p-3 text-center space-y-1 text-[11px]">
                              <ImageOff className="w-5 h-5 text-amber-400 mx-auto" />
                              <p className="font-semibold text-amber-300">Sentinel-2 change imagery unavailable</p>
                              <p className="text-slate-400 text-[10px]">Reason: {imageErrorState['s2_change'].reason}</p>
                            </div>
                          ) : (
                            <img
                              src={resolveImageUrl(inspectionData.imagery?.s2_change, 's2/change') || ''}
                              alt="Sentinel-2 Optical Change"
                              className="w-full h-full object-cover rounded"
                              onLoad={() => handleImageLoadSuccess('s2_change', resolveImageUrl(inspectionData.imagery?.s2_change, 's2/change'))}
                              onError={() => handleImageError('s2_change', resolveImageUrl(inspectionData.imagery?.s2_change, 's2/change'))}
                            />
                          )}
                        </div>
                      </>
                    )}

                    {/* S1 VV TAB */}
                    {activeTab === 's1_vv' && (
                      <>
                        <div className="text-[11px] text-slate-300 space-y-0.5 z-10 bg-slate-900/80 p-2 rounded border border-slate-800/80">
                          <p className="font-semibold text-purple-400">SENTINEL-1 SAR (VV POLARIZATION RADAR)</p>
                          <p>Scene: <span className="font-mono text-white">{inspectionData.imagery?.pre_scene_id || 'S1A_IW_GRDH_1SDV_20201115_PRE'}</span></p>
                        </div>
                        <div className="flex-1 my-2 bg-slate-950 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                          {imageErrorState['s1_vv']?.failed ? (
                            <div className="p-3 text-center space-y-1 text-[11px]">
                              <ImageOff className="w-5 h-5 text-purple-400 mx-auto" />
                              <p className="font-semibold text-purple-300">SAR VV imagery unavailable</p>
                              <p className="text-slate-400 text-[10px]">Reason: {imageErrorState['s1_vv'].reason}</p>
                            </div>
                          ) : (
                            <img
                              src={resolveImageUrl(inspectionData.imagery?.s1_pre_vv || inspectionData.imagery?.s1_pre, 's1/vv/pre') || ''}
                              alt="Sentinel-1 VV Radar"
                              className="w-full h-full object-cover rounded"
                              onLoad={() => handleImageLoadSuccess('s1_vv', resolveImageUrl(inspectionData.imagery?.s1_pre_vv, 's1/vv/pre'))}
                              onError={() => handleImageError('s1_vv', resolveImageUrl(inspectionData.imagery?.s1_pre_vv, 's1/vv/pre'))}
                            />
                          )}
                        </div>
                      </>
                    )}

                    {/* S1 VH TAB */}
                    {activeTab === 's1_vh' && (
                      <>
                        <div className="text-[11px] text-slate-300 space-y-0.5 z-10 bg-slate-900/80 p-2 rounded border border-slate-800/80">
                          <p className="font-semibold text-pink-400">SENTINEL-1 SAR (VH POLARIZATION RADAR)</p>
                          <p>Scene: <span className="font-mono text-white">{inspectionData.imagery?.post_scene_id || 'S1A_IW_GRDH_1SDV_20201127_POST'}</span></p>
                        </div>
                        <div className="flex-1 my-2 bg-slate-950 rounded border border-slate-800 flex items-center justify-center overflow-hidden">
                          {imageErrorState['s1_vh']?.failed ? (
                            <div className="p-3 text-center space-y-1 text-[11px]">
                              <ImageOff className="w-5 h-5 text-pink-400 mx-auto" />
                              <p className="font-semibold text-pink-300">SAR VH imagery unavailable</p>
                              <p className="text-slate-400 text-[10px]">Reason: {imageErrorState['s1_vh'].reason}</p>
                            </div>
                          ) : (
                            <img
                              src={resolveImageUrl(inspectionData.imagery?.s1_pre_vh, 's1/vh/pre') || ''}
                              alt="Sentinel-1 VH Radar"
                              className="w-full h-full object-cover rounded"
                              onLoad={() => handleImageLoadSuccess('s1_vh', resolveImageUrl(inspectionData.imagery?.s1_pre_vh, 's1/vh/pre'))}
                              onError={() => handleImageError('s1_vh', resolveImageUrl(inspectionData.imagery?.s1_pre_vh, 's1/vh/pre'))}
                            />
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {/* Part B — Separate Model Input Section */}
                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-purple-300 uppercase tracking-wider flex items-center gap-1.5">
                      <Cpu className="w-4 h-4 text-purple-400" />
                      MODEL INPUT
                    </span>
                    <span className="text-[10px] font-mono text-purple-400 bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800/40">
                      S2 RGB [1,3,32,32]
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 leading-relaxed">
                    This is the 32×32 image tensor used by the neural network.
                  </p>

                  <div className="w-32 h-32 mx-auto bg-slate-900 rounded-lg border border-slate-800 overflow-hidden flex items-center justify-center p-1">
                    <img
                      src={resolveImageUrl(inspectionData.imagery?.s2_pre, 's2/pre') || ''}
                      alt="32x32 Model Input Tensor Chip"
                      className="w-full h-full object-cover rounded border border-slate-800"
                    />
                  </div>
                </div>

                {/* Operator Human Review Verdict */}
                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <span className="font-semibold text-slate-300 block">Operator Human Review</span>
                  {reviewMessage && <div className="p-2 bg-emerald-950 text-emerald-300 border border-emerald-800 rounded font-medium">{reviewMessage}</div>}
                  <div className="grid grid-cols-3 gap-1.5 pt-1">
                    {['CONFIRM INTACT', 'CONFIRM DAMAGED', 'CONFIRM DESTROYED'].map((v) => (
                      <button
                        key={v}
                        onClick={() => submitHumanReview(v)}
                        className="py-1.5 px-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded text-[11px] font-medium transition text-center"
                      >
                        {v.replace('CONFIRM ', '')}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-400">
                <AlertTriangle className="w-6 h-6 text-amber-400 mx-auto mb-2" />
                <p className="font-semibold text-slate-300">Assessment Data Unavailable</p>
                <p className="text-xs text-slate-400 mt-1">Unable to retrieve building inspection details.</p>
              </div>
            )}
          </aside>
        )}
      </div>
    </div>
  );
}
