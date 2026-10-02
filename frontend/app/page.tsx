"use client";

import React, { useState, useEffect, useRef } from "react";
import dynamic from "next/dynamic";
import {
  Satellite,
  Calendar,
  MapPin,
  Cloud,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  ShieldCheck,
  RefreshCw,
  Search,
  ChevronDown,
  ChevronUp,
  Eye,
  Globe,
  Database,
  RotateCcw,
  RotateCw,
  Building2,
  Compass,
  BarChart3,
  Navigation,
  ShieldAlert
} from "lucide-react";

// Dynamic import Leaflet map for Next.js SSR compatibility
const LeafletMap = dynamic(() => import("../components/ui/leaflet-map"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[500px] bg-neutral-100 rounded-2xl flex flex-col items-center justify-center text-neutral-600 text-xs border border-neutral-300">
      <RefreshCw className="w-8 h-8 text-black animate-spin mb-2" />
      <span>Loading Interactive OpenStreetMap GIS Platform...</span>
    </div>
  ),
});

interface EventRegistryItem {
  event_id: string;
  event_name: string;
  disaster_type: string;
  location: string;
  pre_date: string;
  post_date: string;
  building_count?: number;
  assessment_geojson?: string;
  ground_truth_status?: string;
  accuracy?: string;
  macro_f1?: string;
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
  ground_truth_status?: string;
}

interface GeoJSONFeature {
  type: "Feature";
  id?: string | number;
  properties: {
    building_id?: string;
    osm_id?: number | string;
    prediction?: string;
    damage_prediction?: string;
    ground_truth?: string;
    confidence?: number;
    latitude?: number;
    longitude?: number;
    [key: string]: any;
  };
  geometry: {
    type: "Polygon" | "Point";
    coordinates: any;
  };
}

interface StreetViewResponse {
  available: boolean;
  building_id?: string;
  reason?: string;
  image_url?: string;
  date?: string;
  label?: string;
  pano_id?: string;
  copyright?: string;
  heading?: number;
  pitch?: number;
  fov?: number;
  requested_location?: { latitude: number; longitude: number };
  panorama_location?: { latitude: number; longitude: number };
  panorama_distance_meters?: number;
}

interface SatelliteData {
  before: {
    available: boolean;
    acquisition_date?: string;
    product_id?: string;
    cloud_cover?: number;
    image_url?: string;
    reason?: string;
  };
  after: {
    available: boolean;
    acquisition_date?: string;
    product_id?: string;
    cloud_cover?: number;
    image_url?: string;
    reason?: string;
  };
  difference_url?: string;
}

interface ManualLocationAnalysisResult {
  status: string;
  event_id: string;
  event_name: string;
  event_date: string;
  search_location: { latitude: number; longitude: number };
  exact_building_found: boolean;
  footprint_matched: boolean;
  matched_building_id?: string | null;
  nearest_building?: { building_id: string; distance_meters: number } | null;
  distance_meters?: number | null;
  location_message: string;
  street_view: StreetViewResponse;
  historical_street_view: {
    before: { available: boolean; reason: string; image_url?: string; date?: string };
    after: { available: boolean; reason: string; image_url?: string; date?: string };
  };
  ground_truth: string;
  model_prediction: string;
  model_confidence?: number | null;
  model_status_note?: string | null;
}

export default function TamilNaduDisasterAssessmentApp() {
  // Event Discovery State
  const [events, setEvents] = useState<EventRegistryItem[]>([]);
  const [selectedEventId, setSelectedEventId] = useState<string>("TN_CHENNAI_FLOOD_2021");
  const [eventSummary, setEventSummary] = useState<EventSummary | null>(null);
  const [buildingCollection, setBuildingCollection] = useState<{ type: "FeatureCollection"; features: GeoJSONFeature[] } | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Manual Coordinate Search State
  const [manualLat, setManualLat] = useState<string>("10.583960");
  const [manualLon, setManualLon] = useState<string>("79.712520");
  const [searchQueryLocation, setSearchQueryLocation] = useState<{ lat: number; lon: number } | null>(null);
  const [manualResult, setManualResult] = useState<ManualLocationAnalysisResult | null>(null);
  const [manualSearching, setManualSearching] = useState<boolean>(false);

  // Selected Building State
  const [selectedBuilding, setSelectedBuilding] = useState<GeoJSONFeature | null>(null);
  const [buildingInspection, setBuildingInspection] = useState<any>(null);

  // Google Street View State
  const [streetView, setStreetView] = useState<StreetViewResponse | null>(null);
  const [svLoading, setSvLoading] = useState<boolean>(false);
  const [svHeading, setSvHeading] = useState<number>(0);
  const [svPitch, setSvPitch] = useState<number>(0);
  const [svFov, setSvFov] = useState<number>(90);

  // Copernicus Satellite Imagery State
  const [satellite, setSatellite] = useState<SatelliteData | null>(null);
  const [satLoading, setSatLoading] = useState<boolean>(false);
  const [sliderMode, setSliderMode] = useState<"SPLIT" | "BEFORE" | "AFTER" | "CHANGE">("SPLIT");
  const [sliderPos, setSliderPos] = useState<number>(50);

  // App UI State
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);

  const isDraggingRef = useRef<boolean>(false);
  const FLASK_URL = process.env.NEXT_PUBLIC_FLASK_URL || "http://127.0.0.1:5001";
  const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://127.0.0.1:8000";

  // 1. Fetch Event Registry on Mount
  useEffect(() => {
    const fetchEvents = async () => {
      try {
        const res = await fetch(`${FLASK_URL}/events`);
        if (res.ok) {
          const data = await res.json();
          const list = data.events || [];
          setEvents(list);
          if (list.length > 0) {
            const firstId = list.find((e: EventRegistryItem) => e.event_id === "TN_GAJA_2018")?.event_id || list[0].event_id;
            setSelectedEventId(firstId);
          }
        }
      } catch (e) {
        console.error("Failed to load events registry:", e);
      }
    };

    const checkHealth = async () => {
      try {
        const res = await fetch(`${FASTAPI_URL}/health`);
        if (res.ok) {
          const data = await res.json();
          setIsDemoMode(data.mock_satellite_api ?? false);
        }
      } catch (e) {
        setIsDemoMode(true);
      }
    };

    fetchEvents();
    checkHealth();
  }, []);

  // 2. Load Selected Event Summary & Buildings
  useEffect(() => {
    if (!selectedEventId) return;

    const loadEventData = async () => {
      setLoading(true);
      setError(null);
      setSelectedBuilding(null);
      setManualResult(null);
      setSearchQueryLocation(null);
      setStreetView(null);
      setSatellite(null);

      try {
        // Fetch Summary
        const sumRes = await fetch(`${FLASK_URL}/events/${selectedEventId}/summary`);
        if (sumRes.ok) {
          const sumData = await sumRes.json();
          setEventSummary(sumData);
        }

        // Fetch Buildings GeoJSON
        const bldRes = await fetch(`${FLASK_URL}/events/${selectedEventId}/buildings`);
        if (bldRes.ok) {
          const bldData = await bldRes.json();
          setBuildingCollection(bldData);
          
          // Auto-select first building if available
          if (bldData.features && bldData.features.length > 0) {
            handleSelectBuilding(bldData.features[0]);
          }
        }
      } catch (e: any) {
        setError(`Failed to load event data: ${e.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadEventData();
  }, [selectedEventId]);

  // 3. Handle Building Selection (Workflow A)
  const handleSelectBuilding = async (feature: GeoJSONFeature) => {
    setManualResult(null);
    setSelectedBuilding(feature);
    const props = feature.properties;
    const bId = String(props.building_id || props.osm_id || "BLD_001");

    // Extract Centroid Lat/Lon
    let lat: number | undefined = props.latitude;
    let lon: number | undefined = props.longitude;

    if (lat === undefined || lon === undefined) {
      if (feature.geometry.type === "Point" && Array.isArray(feature.geometry.coordinates)) {
        lon = feature.geometry.coordinates[0];
        lat = feature.geometry.coordinates[1];
      } else if (feature.geometry.type === "Polygon" && Array.isArray(feature.geometry.coordinates)) {
        const ring = feature.geometry.coordinates[0];
        let sumLat = 0, sumLon = 0;
        ring.forEach((c: number[]) => { sumLon += c[0]; sumLat += c[1]; });
        lat = sumLat / ring.length;
        lon = sumLon / ring.length;
      }
    }

    if (lat !== undefined && lon !== undefined) {
      setSearchQueryLocation({ lat, lon });
      fetchStreetView(bId, lat, lon, svHeading, svPitch, svFov);
      fetchSatelliteImagery(lat, lon);
    }

    // Fetch Full Building Inspection Details
    try {
      const inspRes = await fetch(`${FLASK_URL}/building/${bId}`);
      if (inspRes.ok) {
        const inspData = await inspRes.json();
        setBuildingInspection(inspData);
      }
    } catch (e) {
      console.error("Building inspection error:", e);
    }
  };

  // 4. Handle Manual Latitude/Longitude Location Search (Workflow B)
  const handleManualSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const latNum = parseFloat(manualLat);
    const lonNum = parseFloat(manualLon);

    if (isNaN(latNum) || latNum < -90 || latNum > 90) {
      setError("Latitude must be a valid number between -90.0 and 90.0 degrees.");
      return;
    }
    if (isNaN(lonNum) || lonNum < -180 || lonNum > 180) {
      setError("Longitude must be a valid number between -180.0 and 180.0 degrees.");
      return;
    }

    setManualSearching(true);
    setSelectedBuilding(null);
    setSearchQueryLocation({ lat: latNum, lon: lonNum });

    try {
      // Step A: Backend Location Analysis (Event + Coords + Spatial Lookup + StreetView)
      const locRes = await fetch(`${FLASK_URL}/api/location/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_id: selectedEventId,
          latitude: latNum,
          longitude: lonNum
        })
      });

      if (locRes.ok) {
        const locData: ManualLocationAnalysisResult = await locRes.json();
        setManualResult(locData);
        setStreetView(locData.street_view);

        // If exact building footprint matched, select that feature
        if (locData.footprint_matched && locData.matched_building_id && buildingCollection?.features) {
          const matchFeat = buildingCollection.features.find(
            f => String(f.properties.building_id || f.properties.osm_id) === String(locData.matched_building_id)
          );
          if (matchFeat) {
            setSelectedBuilding(matchFeat);
          }
        }
      }

      // Step B: Fetch Satellite Imagery for Searched Location
      await fetchSatelliteImagery(latNum, lonNum);
    } catch (err: any) {
      setError(`Location search failed: ${err.message}`);
    } finally {
      setManualSearching(false);
    }
  };

  // 5. Fetch Google Street View static imagery & metadata
  const fetchStreetView = async (bId: string, lat: number, lon: number, heading: number, pitch: number, fov: number) => {
    setSvLoading(true);
    try {
      const res = await fetch(`${FLASK_URL}/api/buildings/${bId}/streetview?lat=${lat}&lon=${lon}&heading=${heading}&pitch=${pitch}&fov=${fov}`);
      const data = await res.json();
      setStreetView(data);
    } catch (e) {
      setStreetView({ available: false, reason: "Street View service unavailable" });
    } finally {
      setSvLoading(false);
    }
  };

  // Rotate / Tilt Street View
  const handleStreetViewRotate = (deltaHeading: number) => {
    const newHeading = (svHeading + deltaHeading + 360) % 360;
    setSvHeading(newHeading);
    if (selectedBuilding) {
      const props = selectedBuilding.properties;
      const bId = String(props.building_id || props.osm_id || "BLD_001");
      const lat = props.latitude || 13.0827;
      const lon = props.longitude || 80.2707;
      fetchStreetView(bId, lat, lon, newHeading, svPitch, svFov);
    } else if (searchQueryLocation) {
      fetchStreetView("manual_coord", searchQueryLocation.lat, searchQueryLocation.lon, newHeading, svPitch, svFov);
    }
  };

  // 6. Fetch Copernicus Sentinel-2 Before/After Satellite Imagery
  const fetchSatelliteImagery = async (lat: number, lon: number) => {
    setSatLoading(true);
    try {
      const evDate = eventSummary?.pre_date?.split("T")[0] || "2018-11-10";
      const res = await fetch(`${FASTAPI_URL}/api/satellite/before-after`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: lat,
          longitude: lon,
          disaster_date: evDate,
          before_days: 30,
          after_days: 30,
          cloud_cover_max: 30
        })
      });

      if (res.ok) {
        const data = await res.json();
        setSatellite({
          before: data.before || { available: false, reason: "No suitable Copernicus imagery found for this event/location." },
          after: data.after || { available: false, reason: "No suitable Copernicus imagery found for this event/location." },
        });
      } else {
        setSatellite({
          before: { available: false, reason: "No suitable Copernicus imagery found for this event/location." },
          after: { available: false, reason: "No suitable Copernicus imagery found for this event/location." }
        });
      }
    } catch (e) {
      setSatellite({
        before: { available: false, reason: "Copernicus API error" },
        after: { available: false, reason: "Copernicus API error" }
      });
    } finally {
      setSatLoading(false);
    }
  };

  // Slider Mouse Handlers
  const handleMouseDown = () => { isDraggingRef.current = true; };
  const handleMouseUp = () => { isDraggingRef.current = false; };
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  const getDamageBadgeStyle = (status?: string) => {
    switch (status) {
      case "INTACT":
        return "bg-neutral-100 text-neutral-800 border-neutral-300";
      case "DAMAGED":
        return "bg-neutral-300 text-neutral-900 border-neutral-400";
      case "DESTROYED":
        return "bg-black text-white border-black";
      default:
        return "bg-neutral-100 text-neutral-700 border-neutral-200";
    }
  };

  // Filter buildings by search query
  const filteredBuildings = buildingCollection?.features?.filter((f) => {
    if (!searchQuery) return true;
    const bId = String(f.properties.building_id || f.properties.osm_id || "").toLowerCase();
    const pred = (f.properties.prediction || f.properties.damage_prediction || "").toLowerCase();
    return bId.includes(searchQuery.toLowerCase()) || pred.includes(searchQuery.toLowerCase());
  }) || [];

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans flex flex-col selection:bg-black selection:text-white">
      {/* 1. HEADER BAR */}
      <header className="bg-white border-b border-neutral-200 px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 sticky top-0 z-30">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-black text-white">
            <Satellite className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-black uppercase">
              Tamil Nadu Disaster Building-Damage Assessment Platform
            </h1>
            <p className="text-xs text-neutral-600 font-medium">
              Real GIS Map • OpenStreetMap • Google Street View • Copernicus Sentinel-2 • Certified xBD AI Model
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {isDemoMode ? (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-neutral-100 text-neutral-900 border border-neutral-300">
              <ShieldAlert className="w-3.5 h-3.5 mr-1.5 text-black" />
              DEMO MODE
            </span>
          ) : (
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-black text-white border border-black">
              <ShieldCheck className="w-3.5 h-3.5 mr-1.5 text-white" />
              PRODUCTION (REAL APIS)
            </span>
          )}
        </div>
      </header>

      {/* ERROR ALERT */}
      {error && (
        <div className="bg-neutral-100 border-b border-neutral-300 text-neutral-900 px-6 py-3 flex items-center justify-between text-xs font-mono">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-black shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={() => setError(null)} className="font-bold underline hover:text-black">Dismiss</button>
        </div>
      )}

      {/* MAIN LAYOUT: SIDEBAR + MAP + BOTTOM DETAILS PANEL */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* 2. SIDEBAR (EVENT SELECTOR, MANUAL SEARCH, STATISTICS, BUILDING LIST) */}
        <aside className="lg:col-span-3 bg-white border-r border-neutral-200 p-5 space-y-6 flex flex-col justify-between overflow-y-auto max-h-[calc(100vh-73px)]">
          <div className="space-y-6">
            {/* EXISTING EVENT SELECTOR */}
            <div className="space-y-2">
              <label className="block text-xs font-extrabold uppercase tracking-wider text-neutral-700 flex items-center">
                <Globe className="w-4 h-4 text-black mr-1.5" />
                Selected Disaster Event
              </label>
              <select
                value={selectedEventId}
                onChange={(e) => setSelectedEventId(e.target.value)}
                className="w-full bg-neutral-50 border border-neutral-300 rounded-xl px-3 py-2.5 text-sm text-neutral-900 font-medium focus:outline-none focus:border-black transition-colors"
              >
                {events.map((ev) => (
                  <option key={ev.event_id} value={ev.event_id}>
                    {ev.event_name} ({ev.disaster_type})
                  </option>
                ))}
              </select>

              {eventSummary && (
                <div className="text-[11px] text-neutral-600 font-mono flex justify-between bg-neutral-50 px-3 py-1.5 rounded-lg border border-neutral-200">
                  <span>Event Date:</span>
                  <span className="font-bold text-black">{eventSummary.pre_date?.split("T")[0]}</span>
                </div>
              )}
            </div>

            {/* MANUAL LATITUDE / LONGITUDE LOCATION SEARCH PANEL */}
            <div className="bg-neutral-50 rounded-2xl p-4 border border-neutral-200 space-y-3">
              <div className="flex items-center justify-between border-b border-neutral-200 pb-2">
                <h3 className="text-xs font-bold text-black flex items-center">
                  <Navigation className="w-4 h-4 text-black mr-1.5" />
                  Manual Location Search
                </h3>
                <span className="text-[10px] bg-neutral-200 text-black px-2 py-0.5 rounded font-mono font-bold">
                  GIS Target
                </span>
              </div>

              <form onSubmit={handleManualSearch} className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[10px] font-bold text-neutral-600 uppercase mb-1">
                      Latitude (-90 to 90)
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      value={manualLat}
                      onChange={(e) => setManualLat(e.target.value)}
                      placeholder="10.583960"
                      className="w-full bg-white border border-neutral-300 rounded-xl px-2.5 py-1.5 text-xs text-neutral-900 font-mono focus:outline-none focus:border-black"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-[10px] font-bold text-neutral-600 uppercase mb-1">
                      Longitude (-180 to 180)
                    </label>
                    <input
                      type="number"
                      step="0.000001"
                      value={manualLon}
                      onChange={(e) => setManualLon(e.target.value)}
                      placeholder="79.712520"
                      className="w-full bg-white border border-neutral-300 rounded-xl px-2.5 py-1.5 text-xs text-neutral-900 font-mono focus:outline-none focus:border-black"
                      required
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={manualSearching}
                  className="w-full bg-black hover:bg-neutral-800 text-white font-extrabold py-2 px-3 rounded-xl text-xs flex items-center justify-center transition-all disabled:opacity-50"
                >
                  {manualSearching ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                      Searching Location...
                    </>
                  ) : (
                    <>
                      <Search className="w-3.5 h-3.5 mr-1.5" />
                      Search Location
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* EVENT STATISTICS */}
            {eventSummary && (
              <div className="bg-neutral-50 rounded-2xl p-4 border border-neutral-200 space-y-3">
                <div className="flex items-center justify-between border-b border-neutral-200 pb-2">
                  <h3 className="text-xs font-bold text-black flex items-center">
                    <BarChart3 className="w-4 h-4 text-black mr-1.5" />
                    Event Damage Statistics
                  </h3>
                  <span className="text-[10px] bg-neutral-200 text-black px-2 py-0.5 rounded font-mono font-bold">
                    Dataset Derived
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-neutral-100 border border-neutral-300 p-2.5 rounded-xl text-center">
                    <span className="text-[10px] text-neutral-700 uppercase font-bold block">INTACT</span>
                    <span className="text-lg font-black text-black">{eventSummary.intact_count}</span>
                  </div>

                  <div className="bg-neutral-200 border border-neutral-400 p-2.5 rounded-xl text-center">
                    <span className="text-[10px] text-neutral-800 uppercase font-bold block">DAMAGED</span>
                    <span className="text-lg font-black text-black">{eventSummary.damaged_count}</span>
                  </div>

                  <div className="bg-black border border-black p-2.5 rounded-xl text-center">
                    <span className="text-[10px] text-neutral-300 uppercase font-bold block">DESTROYED</span>
                    <span className="text-lg font-black text-white">{eventSummary.destroyed_count}</span>
                  </div>

                  <div className="bg-white border border-neutral-300 p-2.5 rounded-xl text-center">
                    <span className="text-[10px] text-neutral-600 uppercase font-bold block">TOTAL</span>
                    <span className="text-lg font-black text-black">{eventSummary.total_buildings}</span>
                  </div>
                </div>
              </div>
            )}

            {/* BUILDING SEARCH & LIST */}
            <div className="space-y-3">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-3 text-neutral-500" />
                <input
                  type="text"
                  placeholder="Filter building ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full bg-neutral-50 border border-neutral-300 rounded-xl pl-9 pr-3 py-2 text-xs text-neutral-900 focus:outline-none focus:border-black"
                />
              </div>

              <div className="flex items-center space-x-1 bg-neutral-100 p-1 rounded-xl border border-neutral-200">
                {["ALL", "INTACT", "DAMAGED", "DESTROYED"].map((st) => (
                  <button
                    key={st}
                    onClick={() => setFilterStatus(st)}
                    className={`flex-1 py-1 rounded-lg text-[10px] font-extrabold transition-all ${
                      filterStatus === st
                        ? "bg-black text-white shadow-sm"
                        : "text-neutral-600 hover:text-black"
                    }`}
                  >
                    {st}
                  </button>
                ))}
              </div>

              <div className="space-y-2">
                <span className="text-xs font-bold text-neutral-700 uppercase tracking-wider block">
                  Event Buildings ({filteredBuildings.length})
                </span>

                <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1">
                  {filteredBuildings.map((f, idx) => {
                    const bId = String(f.properties.building_id || f.properties.osm_id || `BLD_${idx}`);
                    const pred = f.properties.prediction || f.properties.damage_prediction || "UNKNOWN";
                    const isSel = String(selectedBuilding?.properties?.building_id || selectedBuilding?.properties?.osm_id) === bId;

                    return (
                      <button
                        key={bId}
                        onClick={() => handleSelectBuilding(f)}
                        className={`w-full text-left px-3 py-2 rounded-xl border text-xs flex items-center justify-between transition-all ${
                          isSel
                            ? "bg-neutral-100 border-black text-black font-bold"
                            : "bg-white border-neutral-200 text-neutral-700 hover:bg-neutral-50"
                        }`}
                      >
                        <span className="truncate font-mono">{bId}</span>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getDamageBadgeStyle(pred)}`}>
                          {pred}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-neutral-200 text-[11px] text-neutral-500 font-mono space-y-1">
            <p>Dataset: Tamil Nadu Cyclone & Flood Archive</p>
            <p>Map Base: OpenStreetMap Tiles</p>
          </div>
        </aside>

        {/* 3. MAIN MAP & DISASTER LOCATION ASSESSMENT PANEL */}
        <main className="lg:col-span-9 flex flex-col bg-neutral-50 relative overflow-y-auto max-h-[calc(100vh-73px)]">
          {/* MAIN INTERACTIVE MAP */}
          <div className="w-full h-[520px] relative border-b border-neutral-200 shrink-0">
            <LeafletMap
              data={buildingCollection}
              selectedFeature={selectedBuilding}
              onSelectBuilding={handleSelectBuilding}
              filterStatus={filterStatus}
              searchQueryLocation={searchQueryLocation}
            />

            {/* MAP OVERLAY BADGE */}
            <div className="absolute top-4 left-4 z-[1000] bg-white border border-neutral-200 px-3 py-2 rounded-xl text-xs space-y-0.5 shadow-sm">
              <span className="font-bold text-black block flex items-center">
                <Globe className="w-3.5 h-3.5 mr-1 text-black" />
                {eventSummary?.event_name || "Tamil Nadu GIS Map"}
              </span>
              <span className="text-[10px] text-neutral-600 font-mono block">
                OPENSTREETMAP (Geographic Base Layer)
              </span>
            </div>
          </div>

          {/* 4. DISASTER LOCATION ASSESSMENT RESULT PANEL */}
          {(manualResult || selectedBuilding) ? (
            <div className="p-6 space-y-6 bg-neutral-50 flex-1">
              {/* HEADER INFO BAR */}
              <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-neutral-200 pb-4 gap-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <MapPin className="w-5 h-5 text-black" />
                    <h2 className="text-lg font-extrabold text-black font-mono tracking-wide">
                      DISASTER LOCATION ASSESSMENT
                    </h2>
                  </div>
                  <div className="text-xs text-neutral-600 mt-1 space-x-3 font-mono">
                    <span>Event: <strong className="text-black font-semibold">{manualResult?.event_name || eventSummary?.event_name}</strong></span>
                    <span>•</span>
                    <span>Event date: <strong className="text-neutral-900">{manualResult?.event_date || eventSummary?.pre_date?.split("T")[0]}</strong></span>
                    <span>•</span>
                    <span>Requested coordinates: <strong className="text-black">{searchQueryLocation ? `${searchQueryLocation.lat.toFixed(6)}, ${searchQueryLocation.lon.toFixed(6)}` : "Dataset Location"}</strong></span>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <span className="text-[10px] text-neutral-500 uppercase font-bold block">GROUND TRUTH</span>
                    <span className={`px-3 py-1 rounded-xl text-xs font-black border ${getDamageBadgeStyle(manualResult ? manualResult.ground_truth : (selectedBuilding?.properties?.ground_truth || selectedBuilding?.properties?.prediction))}`}>
                      {manualResult ? manualResult.ground_truth : (selectedBuilding?.properties?.ground_truth || selectedBuilding?.properties?.prediction || "UNAVAILABLE")}
                    </span>
                  </div>

                  <div className="text-right">
                    <span className="text-[10px] text-neutral-500 uppercase font-bold block">MODEL PREDICTION</span>
                    <span className={`px-3 py-1 rounded-xl text-xs font-black border ${getDamageBadgeStyle(manualResult ? manualResult.model_prediction : selectedBuilding?.properties?.prediction)}`}>
                      {manualResult ? manualResult.model_prediction : (selectedBuilding?.properties?.prediction || "unavailable")}
                    </span>
                  </div>
                </div>
              </div>

              {/* BUILDING SECTION */}
              <div className="bg-white rounded-2xl p-4 border border-neutral-200 text-xs font-mono space-y-2">
                <span className="text-[11px] font-bold text-black uppercase tracking-wider block">BUILDING</span>
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 text-neutral-800">
                  <div className="flex items-center space-x-4">
                    <span>
                      Exact building: <strong className="text-black">
                        {manualResult ? (manualResult.exact_building_found ? `Found (${manualResult.matched_building_id})` : "Not Found") : `Found (${selectedBuilding?.properties?.building_id || selectedBuilding?.properties?.osm_id})`}
                      </strong>
                    </span>

                    {manualResult?.nearest_building && (
                      <span>
                        Nearest building: <strong className="text-black">{manualResult.nearest_building.building_id}</strong> (Distance: {manualResult.nearest_building.distance_meters} m)
                      </span>
                    )}
                  </div>

                  {manualResult?.nearest_building && (
                    <button
                      onClick={() => {
                        if (buildingCollection?.features) {
                          const matchFeat = buildingCollection.features.find(
                            f => String(f.properties.building_id || f.properties.osm_id) === String(manualResult.nearest_building?.building_id)
                          );
                          if (matchFeat) handleSelectBuilding(matchFeat);
                        }
                      }}
                      className="px-3 py-1 rounded-xl bg-white hover:bg-neutral-100 text-black border border-neutral-300 text-[11px] font-bold transition-colors shrink-0"
                    >
                      Select Nearest Building
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-neutral-600">{manualResult?.location_message || "Exact building polygon retrieved from spatial database."}</p>
              </div>

              {/* GRID: GOOGLE STREET VIEW vs COPERNICUS SENTINEL */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* GOOGLE STREET VIEW PANEL */}
                <div className="lg:col-span-5 bg-white border border-neutral-200 rounded-3xl p-5 space-y-4 shadow-sm flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-extrabold text-black flex items-center">
                        <Compass className="w-4 h-4 text-black mr-2" />
                        GOOGLE STREET VIEW: Current Street-Level Context
                      </h3>
                      {streetView?.available ? (
                        <span className="text-[10px] bg-neutral-100 text-neutral-900 px-2 py-0.5 rounded-lg border border-neutral-300 font-bold">
                          CURRENT STREET VIEW ({streetView.date || "Captured"})
                        </span>
                      ) : (
                        <span className="text-[10px] bg-neutral-200 text-neutral-700 px-2 py-0.5 rounded-lg border border-neutral-300 font-bold">
                          Unavailable
                        </span>
                      )}
                    </div>

                    <div className="aspect-video bg-neutral-100 rounded-2xl overflow-hidden border border-neutral-200 relative flex items-center justify-center">
                      {svLoading ? (
                        <div className="text-center space-y-2 text-neutral-600 text-xs">
                          <RefreshCw className="w-6 h-6 animate-spin mx-auto text-black" />
                          <span>Checking Google Street View API...</span>
                        </div>
                      ) : streetView?.available && streetView.image_url ? (
                        <img
                          src={streetView.image_url}
                          alt="Google Street View Real Imagery"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="p-6 text-center space-y-2">
                          <AlertTriangle className="w-8 h-8 text-neutral-500 mx-auto" />
                          <p className="text-xs font-bold text-neutral-800">
                            Street View imagery is not available at this location.
                          </p>
                          <p className="text-[10px] text-neutral-500">
                            No panorama recorded by Google Street View near coordinates.
                          </p>
                        </div>
                      )}
                    </div>

                    {/* STREET VIEW PANORAMA METADATA & DISTANCE */}
                    {streetView?.available && (
                      <div className="bg-neutral-50 rounded-2xl p-3 border border-neutral-200 text-[11px] text-neutral-800 space-y-1 font-mono">
                        <p className="flex justify-between">
                          <span className="text-neutral-500">Classification:</span>
                          <span className="font-bold text-black">CURRENT STREET VIEW</span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-neutral-500">Captured:</span>
                          <span className="font-bold text-neutral-900">{streetView.date || "Current"}</span>
                        </p>
                        {searchQueryLocation && (
                          <p className="flex justify-between">
                            <span className="text-neutral-500">Requested:</span>
                            <span className="font-bold text-neutral-900">{searchQueryLocation.lat.toFixed(6)}, {searchQueryLocation.lon.toFixed(6)}</span>
                          </p>
                        )}
                        {streetView.panorama_location && (
                          <p className="flex justify-between">
                            <span className="text-neutral-500">Street View panorama:</span>
                            <span className="font-bold text-neutral-900">{streetView.panorama_location.latitude.toFixed(6)}, {streetView.panorama_location.longitude.toFixed(6)}</span>
                          </p>
                        )}
                        {streetView.panorama_distance_meters !== undefined && (
                          <p className="flex justify-between">
                            <span className="text-neutral-500">Distance:</span>
                            <span className="font-bold text-black">{streetView.panorama_distance_meters} m</span>
                          </p>
                        )}
                      </div>
                    )}

                    {/* HISTORICAL STREET VIEW NOTICE */}
                    <div className="bg-white rounded-2xl p-3 border border-neutral-200 text-[11px] text-neutral-700 space-y-1 font-mono">
                      <span className="text-[10px] font-bold text-neutral-500 uppercase block mb-1">HISTORICAL STREET VIEW</span>
                      <p className="flex justify-between"><span className="text-neutral-500">Historical Before:</span> <span className="text-black font-bold">Unavailable through configured API</span></p>
                      <p className="flex justify-between"><span className="text-neutral-500">Historical After:</span> <span className="text-black font-bold">Unavailable through configured API</span></p>
                    </div>
                  </div>

                  {streetView?.available && (
                    <div className="flex items-center justify-between pt-2 border-t border-neutral-200">
                      <span className="text-[11px] text-neutral-600 font-mono">
                        Heading: {svHeading}°
                      </span>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleStreetViewRotate(-45)}
                          className="p-2 bg-white hover:bg-neutral-100 rounded-xl border border-neutral-300 text-black text-xs flex items-center"
                          title="Rotate Left 45°"
                        >
                          <RotateCcw className="w-3.5 h-3.5 mr-1" /> Left
                        </button>
                        <button
                          onClick={() => handleStreetViewRotate(45)}
                          className="p-2 bg-white hover:bg-neutral-100 rounded-xl border border-neutral-300 text-black text-xs flex items-center"
                          title="Rotate Right 45°"
                        >
                          <RotateCw className="w-3.5 h-3.5 mr-1" /> Right
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* COPERNICUS SENTINEL SATELLITE BEFORE/AFTER */}
                <div className="lg:col-span-7 bg-white border border-neutral-200 rounded-3xl p-5 space-y-4 shadow-sm flex flex-col justify-between">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-extrabold text-black flex items-center">
                        <Eye className="w-4 h-4 text-black mr-2" />
                        Copernicus Sentinel — Before / After
                      </h3>
                      <div className="flex items-center space-x-1 bg-neutral-100 p-1 rounded-xl border border-neutral-200 text-[10px]">
                        <button
                          onClick={() => setSliderMode("SPLIT")}
                          className={`px-2.5 py-1 rounded-lg font-bold ${sliderMode === "SPLIT" ? "bg-black text-white" : "text-neutral-600"}`}
                        >
                          Split Slider
                        </button>
                        <button
                          onClick={() => setSliderMode("BEFORE")}
                          className={`px-2.5 py-1 rounded-lg font-bold ${sliderMode === "BEFORE" ? "bg-black text-white" : "text-neutral-600"}`}
                        >
                          Before
                        </button>
                        <button
                          onClick={() => setSliderMode("AFTER")}
                          className={`px-2.5 py-1 rounded-lg font-bold ${sliderMode === "AFTER" ? "bg-black text-white" : "text-neutral-600"}`}
                        >
                          After
                        </button>
                      </div>
                    </div>

                    {/* SATELLITE DISPLAY */}
                    {satLoading ? (
                      <div className="aspect-video bg-neutral-100 rounded-2xl flex flex-col items-center justify-center text-neutral-600 text-xs border border-neutral-200">
                        <RefreshCw className="w-6 h-6 animate-spin text-black mb-2" />
                        <span>Searching Copernicus Data Space Ecosystem...</span>
                      </div>
                    ) : satellite?.before?.available && satellite?.after?.available ? (
                      <div
                        className="relative w-full aspect-video bg-neutral-100 rounded-2xl overflow-hidden border border-neutral-300 select-none cursor-col-resize"
                        onMouseDown={handleMouseDown}
                        onMouseUp={handleMouseUp}
                        onMouseMove={handleMouseMove}
                      >
                        <img src={satellite.after.image_url} alt="After Sentinel Satellite" className="w-full h-full object-cover" />
                        <div
                          className="absolute inset-0 h-full overflow-hidden border-r-2 border-black"
                          style={{ width: `${sliderPos}%` }}
                        >
                          <img src={satellite.before.image_url} alt="Before Sentinel Satellite" className="w-full h-full object-cover" style={{ width: "100%", height: "100%" }} />
                        </div>
                        <div
                          className="absolute top-0 bottom-0 w-1 bg-black cursor-col-resize"
                          style={{ left: `${sliderPos}%` }}
                        />
                      </div>
                    ) : (
                      <div className="aspect-video bg-neutral-100 rounded-2xl p-6 flex flex-col items-center justify-center text-center space-y-2 border border-neutral-200">
                        <AlertTriangle className="w-8 h-8 text-neutral-500" />
                        <p className="text-xs font-bold text-neutral-800">
                          {satellite?.before?.reason || satellite?.after?.reason || "No suitable Copernicus Sentinel image found for this event/location."}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* SATELLITE METADATA GRID */}
                  {satellite?.before?.available && satellite?.after?.available && (
                    <div className="grid grid-cols-2 gap-3 text-[11px] font-mono">
                      <div className="bg-neutral-50 rounded-2xl p-3 border border-neutral-200 space-y-1">
                        <span className="text-xs font-bold text-black block mb-1">COPERNICUS SENTINEL — BEFORE DISASTER</span>
                        <p className="flex justify-between"><span className="text-neutral-500">Acquisition:</span><span className="font-bold text-neutral-900">{satellite.before.acquisition_date?.split("T")[0]}</span></p>
                        <p className="flex justify-between"><span className="text-neutral-500">Cloud:</span><span className="font-bold text-neutral-900">{satellite.before.cloud_cover}%</span></p>
                        <p className="truncate text-[10px] text-neutral-600" title={satellite.before.product_id}>Product: {satellite.before.product_id}</p>
                      </div>

                      <div className="bg-neutral-50 rounded-2xl p-3 border border-neutral-200 space-y-1">
                        <span className="text-xs font-bold text-black block mb-1">COPERNICUS SENTINEL — AFTER DISASTER</span>
                        <p className="flex justify-between"><span className="text-neutral-500">Acquisition:</span><span className="font-bold text-neutral-900">{satellite.after.acquisition_date?.split("T")[0]}</span></p>
                        <p className="flex justify-between"><span className="text-neutral-500">Cloud:</span><span className="font-bold text-neutral-900">{satellite.after.cloud_cover}%</span></p>
                        <p className="truncate text-[10px] text-neutral-600" title={satellite.after.product_id}>Product: {satellite.after.product_id}</p>
                      </div>
                    </div>
                  )}

                  {/* DAMAGE ASSESSMENT / MODEL STATUS NOTE */}
                  <div className="bg-neutral-50 rounded-2xl p-3 border border-neutral-200 text-[11px] text-neutral-700 space-y-1 font-mono">
                    <span className="text-[11px] font-bold text-black uppercase tracking-wider block mb-1">DAMAGE ASSESSMENT</span>
                    <p className="flex justify-between"><span className="text-neutral-500">Ground Truth:</span> <span className="font-bold text-black">{manualResult ? manualResult.ground_truth : (selectedBuilding?.properties?.ground_truth || selectedBuilding?.properties?.prediction || "UNAVAILABLE")}</span></p>
                    <p className="flex justify-between"><span className="text-neutral-500">Model Prediction:</span> <span className="font-bold text-black">{manualResult ? manualResult.model_prediction : (selectedBuilding?.properties?.prediction || "unavailable")}</span></p>
                    <p className="flex justify-between"><span className="text-neutral-500">Confidence:</span> <span className="font-bold text-black">{manualResult?.model_confidence ? `${(manualResult.model_confidence * 100).toFixed(1)}%` : "unavailable"}</span></p>
                  </div>

                </div>
              </div>
            </div>
          ) : (
            <div className="p-12 text-center text-neutral-500 space-y-2">
              <Building2 className="w-10 h-10 mx-auto text-neutral-400" />
              <p className="text-sm font-bold text-neutral-700">Select an event building or enter manual coordinates to run location assessment.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
