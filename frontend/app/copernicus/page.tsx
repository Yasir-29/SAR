"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
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
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  Search,
  Sparkles,
  ChevronLeft,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  Sliders,
  Eye,
  FileJson,
  FileText
} from "lucide-react";

interface ImageryMeta {
  date: string;
  cloud_cover: number;
  product_id: string;
  source: string;
  image_url: string;
}

interface AnalysisResult {
  request_id: string;
  location: {
    latitude: number;
    longitude: number;
  };
  before: ImageryMeta;
  after: ImageryMeta;
  comparison: {
    difference_image_url: string;
  };
  model: {
    class: "INTACT" | "DAMAGED" | "DESTROYED";
    confidence: number;
  };
}

const LOADING_STAGES = [
  "Searching satellite imagery...",
  "Downloading before image...",
  "Downloading after image...",
  "Creating comparison...",
  "Running damage assessment...",
  "Analysis complete."
];

export default function CopernicusAnalysisPage() {
  const [latitude, setLatitude] = useState<string>("11.0168");
  const [longitude, setLongitude] = useState<string>("76.9558");
  const [disasterDate, setDisasterDate] = useState<string>("2025-01-15");
  const [beforeDays, setBeforeDays] = useState<number>(30);
  const [afterDays, setAfterDays] = useState<number>(30);
  const [cloudCoverMax, setCloudCoverMax] = useState<number>(30);

  const [loading, setLoading] = useState<boolean>(false);
  const [loadingStageIndex, setLoadingStageIndex] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // View state
  const [viewMode, setViewMode] = useState<"SPLIT" | "SIDE_BY_SIDE" | "BEFORE" | "AFTER" | "CHANGE">("SPLIT");
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const isDragging = useRef<boolean>(false);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<any>(null);

  // Split view dragging handler
  const handleMouseDown = () => {
    isDragging.current = true;
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPos(percent);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLDivElement>) => {
    if (!e.touches[0]) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
    const percent = (x / rect.width) * 100;
    setSliderPos(percent);
  };

  // Render Leaflet Map dynamically
  useEffect(() => {
    if (typeof window === "undefined") return;
    const containerEl = mapContainerRef.current;
    if (!result || !containerEl) return;

    const lat = result.location.latitude;
    const lon = result.location.longitude;

    import("leaflet").then((L) => {
      // Fix leaflet marker icon url issues in Next.js
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });

      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
      }

      const map = L.map(containerEl).setView([lat, lon], 15);
      leafletMapRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors | Copernicus CDSE",
        maxZoom: 19,
      }).addTo(map);

      // Marker at center
      L.marker([lat, lon])
        .addTo(map)
        .bindPopup(`<b>Target AOI Coordinate</b><br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`)
        .openPopup();

      // Bounding box (~500m x 500m)
      const latDelta = 250 / 111320;
      const lonDelta = 250 / (111320 * Math.cos((lat * Math.PI) / 180));
      const bounds: [number, number][] = [
        [lat - latDelta, lon - lonDelta],
        [lat + latDelta, lon + lonDelta],
      ];

      L.rectangle(bounds, {
        color: "#38bdf8",
        weight: 2,
        fillColor: "#0284c7",
        fillOpacity: 0.15,
      }).addTo(map);
    });

    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
    };
  }, [result]);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    const latNum = parseFloat(latitude);
    const lonNum = parseFloat(longitude);

    if (isNaN(latNum) || latNum < -90 || latNum > 90) {
      setError("Latitude must be a valid coordinate between -90 and 90.");
      return;
    }
    if (isNaN(lonNum) || lonNum < -180 || lonNum > 180) {
      setError("Longitude must be a valid coordinate between -180 and 180.");
      return;
    }
    if (!disasterDate) {
      setError("Please select a valid disaster date.");
      return;
    }

    setLoading(true);
    setLoadingStageIndex(0);

    const interval = setInterval(() => {
      setLoadingStageIndex((prev) => {
        if (prev < LOADING_STAGES.length - 2) {
          return prev + 1;
        }
        return prev;
      });
    }, 850);

    try {
      const backendUrl = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://127.0.0.1:8000";
      const res = await fetch(`${backendUrl}/api/disaster/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          latitude: latNum,
          longitude: lonNum,
          disaster_date: disasterDate,
          before_days: beforeDays,
          after_days: afterDays,
          cloud_cover_max: cloudCoverMax,
        }),
      });

      clearInterval(interval);

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        let errMsg = errData.detail || `Server returned error status ${res.status}`;
        if (res.status === 404) {
          errMsg = "No suitable pre-disaster or post-disaster imagery found for the requested criteria.";
        }
        throw new Error(errMsg);
      }

      setLoadingStageIndex(LOADING_STAGES.length - 1);
      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (err: any) {
      clearInterval(interval);
      setError(err.message || "Failed to retrieve imagery or run assessment.");
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDownloadComparisonBundle = () => {
    if (!result) return;
    
    // Download Before Image
    downloadFile(result.before.image_url, `${result.request_id}_BEFORE.png`);

    // Download After Image
    setTimeout(() => {
      downloadFile(result.after.image_url, `${result.request_id}_AFTER.png`);
    }, 300);

    // Download Difference Image
    setTimeout(() => {
      downloadFile(result.comparison.difference_image_url, `${result.request_id}_DIFFERENCE.png`);
    }, 600);

    // Download Metadata JSON
    setTimeout(() => {
      const jsonString = `data:text/json;charset=utf-8,${encodeURIComponent(
        JSON.stringify(result, null, 2)
      )}`;
      downloadFile(jsonString, `${result.request_id}_METADATA.json`);
    }, 900);
  };

  const getBadgeStyle = (cls: string) => {
    switch (cls) {
      case "INTACT":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-emerald-500/10";
      case "DAMAGED":
        return "bg-amber-500/20 text-amber-400 border-amber-500/50 shadow-amber-500/10";
      case "DESTROYED":
        return "bg-rose-500/20 text-rose-400 border-rose-500/50 shadow-rose-500/10";
      default:
        return "bg-sky-500/20 text-sky-400 border-sky-500/50 shadow-sky-500/10";
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans p-4 md:p-8">
      {/* Top Header */}
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between mb-3">
          <Link
            href="/"
            className="inline-flex items-center text-xs font-semibold text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ChevronLeft className="w-4 h-4 mr-1" /> Return to Main Dashboard
          </Link>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/30">
              <ShieldCheck className="w-3.5 h-3.5 mr-1.5" />
              Copernicus CDSE API (Sentinel-2 L2A)
            </span>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition-colors"
            >
              API Swagger
            </a>
          </div>
        </div>

        <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-sky-400 via-indigo-300 to-purple-400">
              Disaster Damage Assessment
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Copernicus Sentinel-2 Satellite Optical Imagery & AI Damage Model Inference Dashboard
            </p>
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="max-w-7xl mx-auto space-y-8">
        {/* INPUT CONTROL PANEL */}
        <section className="bg-slate-900/80 backdrop-blur border border-slate-800 rounded-2xl p-6 shadow-xl">
          <form onSubmit={handleAnalyze} className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Latitude
              </label>
              <div className="relative">
                <MapPin className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
                <input
                  type="number"
                  step="0.0001"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                  className="w-full bg-slate-800/90 border border-slate-700 rounded-xl pl-9 pr-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-sky-500 transition-colors"
                  placeholder="11.0168"
                  required
                />
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Longitude
              </label>
              <div className="relative">
                <MapPin className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
                <input
                  type="number"
                  step="0.0001"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                  className="w-full bg-slate-800/90 border border-slate-700 rounded-xl pl-9 pr-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-sky-500 transition-colors"
                  placeholder="76.9558"
                  required
                />
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Disaster Date
              </label>
              <div className="relative">
                <Calendar className="w-4 h-4 absolute left-3 top-3 text-slate-500" />
                <input
                  type="date"
                  value={disasterDate}
                  onChange={(e) => setDisasterDate(e.target.value)}
                  className="w-full bg-slate-800/90 border border-slate-700 rounded-xl pl-9 pr-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-sky-500 transition-colors"
                  required
                />
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Before Window (Days)
              </label>
              <input
                type="number"
                min="1"
                max="90"
                value={beforeDays}
                onChange={(e) => setBeforeDays(parseInt(e.target.value) || 30)}
                className="w-full bg-slate-800/90 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-slate-100 focus:outline-none focus:border-sky-500 transition-colors"
              />
            </div>

            <div className="md:col-span-2">
              <label className="block text-xs font-semibold text-slate-400 mb-1">
                Cloud Cover Max ({cloudCoverMax}%)
              </label>
              <input
                type="range"
                min="0"
                max="100"
                value={cloudCoverMax}
                onChange={(e) => setCloudCoverMax(parseInt(e.target.value) || 30)}
                className="w-full accent-sky-500 mt-2"
              />
            </div>

            <div className="md:col-span-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-bold py-2.5 px-4 rounded-xl shadow-lg shadow-sky-500/20 flex items-center justify-center transition-all disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </form>

          {/* Presets */}
          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center space-x-3 text-xs">
            <span className="text-slate-500 font-medium">Quick Disaster Presets:</span>
            <button
              type="button"
              onClick={() => {
                setLatitude("13.0429");
                setLongitude("80.2486");
                setDisasterDate("2021-11-10");
              }}
              className="bg-slate-800 hover:bg-slate-700 text-sky-300 px-2.5 py-1 rounded-lg border border-slate-700 transition-colors"
            >
              Chennai Flood 2021
            </button>
            <button
              type="button"
              onClick={() => {
                setLatitude("10.7656");
                setLongitude("79.8424");
                setDisasterDate("2018-11-15");
              }}
              className="bg-slate-800 hover:bg-slate-700 text-amber-300 px-2.5 py-1 rounded-lg border border-slate-700 transition-colors"
            >
              Nagapattinam (Gaja)
            </button>
            <button
              type="button"
              onClick={() => {
                setLatitude("13.0827");
                setLongitude("80.2707");
                setDisasterDate("2016-12-12");
              }}
              className="bg-slate-800 hover:bg-slate-700 text-indigo-300 px-2.5 py-1 rounded-lg border border-slate-700 transition-colors"
            >
              Vardah 2016
            </button>
          </div>
        </section>

        {/* ERROR DISPLAY */}
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-2xl p-5 flex items-start space-x-3 shadow-lg">
            <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-sm">Disaster Analysis Notice</h4>
              <p className="text-xs text-rose-200 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* LOADING PROGRESSIVE DISPLAY */}
        {loading && (
          <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-2xl p-10 text-center space-y-4 shadow-2xl">
            <div className="inline-flex p-4 rounded-full bg-sky-500/10 text-sky-400 animate-pulse border border-sky-500/20">
              <Satellite className="w-10 h-10" />
            </div>
            <h3 className="text-xl font-extrabold text-slate-200 tracking-tight">
              {LOADING_STAGES[loadingStageIndex]}
            </h3>
            <div className="w-full bg-slate-800 rounded-full h-2.5 max-w-lg mx-auto overflow-hidden">
              <div
                className="bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500 h-full transition-all duration-500"
                style={{
                  width: `${((loadingStageIndex + 1) / LOADING_STAGES.length) * 100}%`,
                }}
              />
            </div>
            <p className="text-xs text-slate-400">
              Executing STAC search, cropping Sentinel-2 L2A optical chips, and invoking model inference...
            </p>
          </div>
        )}

        {/* RESULTS SECTION */}
        {result && !loading && (
          <div className="space-y-8">
            {/* VIEW MODE TOGGLE TOOLBAR */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 flex flex-col md:flex-row items-center justify-between gap-4 shadow-xl">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mr-2">
                  View Mode:
                </span>
                <button
                  type="button"
                  onClick={() => setViewMode("SPLIT")}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                    viewMode === "SPLIT"
                      ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  SPLIT VIEW
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("SIDE_BY_SIDE")}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                    viewMode === "SIDE_BY_SIDE"
                      ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  SIDE-BY-SIDE
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("BEFORE")}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                    viewMode === "BEFORE"
                      ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  BEFORE
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("AFTER")}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                    viewMode === "AFTER"
                      ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  AFTER
                </button>
                <button
                  type="button"
                  onClick={() => setViewMode("CHANGE")}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                    viewMode === "CHANGE"
                      ? "bg-sky-500 text-white shadow-md shadow-sky-500/30"
                      : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                  }`}
                >
                  CHANGE
                </button>
              </div>

              {/* Zoom & Export Actions */}
              <div className="flex items-center space-x-3">
                <div className="flex items-center bg-slate-800 rounded-xl p-1 border border-slate-700">
                  <button
                    type="button"
                    onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 2.5))}
                    className="p-1.5 text-slate-300 hover:text-white transition-colors"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <span className="text-xs font-mono text-slate-400 px-2">
                    {Math.round(zoomLevel * 100)}%
                  </span>
                  <button
                    type="button"
                    onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 1))}
                    className="p-1.5 text-slate-300 hover:text-white transition-colors"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setZoomLevel(1)}
                    className="p-1.5 text-slate-300 hover:text-white transition-colors border-l border-slate-700 ml-1 pl-2"
                    title="Reset Zoom"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                <button
                  type="button"
                  onClick={handleDownloadComparisonBundle}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1.5 rounded-xl text-xs font-bold flex items-center shadow-lg shadow-emerald-600/20 transition-all"
                >
                  <Download className="w-3.5 h-3.5 mr-1.5" />
                  Download Bundle
                </button>
              </div>
            </div>

            {/* INTERACTIVE DRAGGABLE SPLIT VIEW */}
            {viewMode === "SPLIT" && (
              <section className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-extrabold text-slate-100 flex items-center">
                    <Sliders className="w-5 h-5 text-sky-400 mr-2" />
                    Interactive Split View Comparison
                  </h2>
                  <span className="text-xs text-slate-400">
                    Drag the vertical slider to compare Before & After imagery pixel-for-pixel
                  </span>
                </div>

                <div
                  className="relative w-full aspect-video md:aspect-[21/9] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 select-none cursor-col-resize"
                  onMouseDown={handleMouseDown}
                  onMouseUp={handleMouseUp}
                  onMouseMove={handleMouseMove}
                  onTouchMove={handleTouchMove}
                >
                  {/* AFTER Image (Full underneath) */}
                  <div className="absolute inset-0 w-full h-full">
                    <img
                      src={result.after.image_url}
                      alt="After Disaster Imagery"
                      className="w-full h-full object-cover"
                      style={{ transform: `scale(${zoomLevel})` }}
                    />
                    <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur text-indigo-300 px-3 py-1 rounded-xl text-xs font-bold border border-indigo-500/30">
                      AFTER DISASTER ({new Date(result.after.date).toLocaleDateString()})
                    </div>
                  </div>

                  {/* BEFORE Image (Clipped overlay) */}
                  <div
                    className="absolute inset-0 h-full overflow-hidden border-r-2 border-white shadow-2xl"
                    style={{ width: `${sliderPos}%` }}
                  >
                    <img
                      src={result.before.image_url}
                      alt="Before Disaster Imagery"
                      className="w-full h-full object-cover"
                      style={{
                        width: "100%",
                        height: "100%",
                        transform: `scale(${zoomLevel})`,
                      }}
                    />
                    <div className="absolute top-4 left-4 bg-slate-900/80 backdrop-blur text-sky-300 px-3 py-1 rounded-xl text-xs font-bold border border-sky-500/30">
                      BEFORE DISASTER ({new Date(result.before.date).toLocaleDateString()})
                    </div>
                  </div>

                  {/* Handle Slider Knob */}
                  <div
                    className="absolute top-0 bottom-0 w-1 bg-white cursor-col-resize shadow-[0_0_15px_rgba(255,255,255,0.8)]"
                    style={{ left: `${sliderPos}%` }}
                  >
                    <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white text-slate-900 flex items-center justify-center shadow-2xl font-bold text-xs">
                      ↔
                    </div>
                  </div>
                </div>
              </section>
            )}

            {/* BEFORE VS AFTER DISASTER SECTION (Side-by-Side or Selected Single View) */}
            {(viewMode === "SIDE_BY_SIDE" || viewMode === "BEFORE" || viewMode === "AFTER") && (
              <section className="space-y-4">
                <h2 className="text-xl font-extrabold text-slate-100 flex items-center">
                  <Eye className="w-5 h-5 text-sky-400 mr-2" />
                  Before vs After Disaster
                </h2>

                <div
                  className={`grid gap-6 ${
                    viewMode === "SIDE_BY_SIDE"
                      ? "grid-cols-1 md:grid-cols-2"
                      : "grid-cols-1 max-w-2xl mx-auto"
                  }`}
                >
                  {/* BEFORE PANEL */}
                  {(viewMode === "SIDE_BY_SIDE" || viewMode === "BEFORE") && (
                    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm font-extrabold text-sky-400 uppercase tracking-wider">
                            BEFORE DISASTER
                          </span>
                          <span className="text-xs bg-slate-800 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-700">
                            Cloud: {result.before.cloud_cover}%
                          </span>
                        </div>

                        <div className="aspect-square bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative group">
                          <img
                            src={result.before.image_url}
                            alt="BEFORE DISASTER Satellite Imagery"
                            className="w-full h-full object-cover transition-transform duration-300"
                            style={{ transform: `scale(${zoomLevel})` }}
                          />
                        </div>
                      </div>

                      <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800/80 text-xs text-slate-300 space-y-1.5 font-mono">
                        <p className="flex justify-between">
                          <span className="text-slate-500">Source:</span>
                          <span className="font-semibold text-slate-200">
                            {result.before.source}
                          </span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-slate-500">Acquisition Date:</span>
                          <span className="font-semibold text-slate-200">
                            {new Date(result.before.date).toLocaleDateString("en-US", {
                              day: "2-digit",
                              month: "short",
                              year: "numeric",
                            })}
                          </span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-slate-500">Cloud Coverage:</span>
                          <span className="font-semibold text-slate-200">
                            {result.before.cloud_cover}%
                          </span>
                        </p>
                        <p className="flex justify-between truncate" title={result.before.product_id}>
                          <span className="text-slate-500">Product ID:</span>
                          <span className="truncate max-w-[200px] text-slate-400">
                            {result.before.product_id}
                          </span>
                        </p>
                      </div>
                    </div>
                  )}

                  {/* AFTER PANEL */}
                  {(viewMode === "SIDE_BY_SIDE" || viewMode === "AFTER") && (
                    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-sm font-extrabold text-indigo-400 uppercase tracking-wider">
                            AFTER DISASTER
                          </span>
                          <span className="text-xs bg-slate-800 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-700">
                            Cloud: {result.after.cloud_cover}%
                          </span>
                        </div>

                        <div className="aspect-square bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative group">
                          <img
                            src={result.after.image_url}
                            alt="AFTER DISASTER Satellite Imagery"
                            className="w-full h-full object-cover transition-transform duration-300"
                            style={{ transform: `scale(${zoomLevel})` }}
                          />
                        </div>
                      </div>

                      <div className="bg-slate-950/80 rounded-xl p-3 border border-slate-800/80 text-xs text-slate-300 space-y-1.5 font-mono">
                        <p className="flex justify-between">
                          <span className="text-slate-500">Source:</span>
                          <span className="font-semibold text-slate-200">
                            {result.after.source}
                          </span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-slate-500">Acquisition Date:</span>
                          <span className="font-semibold text-slate-200">
                            {new Date(result.after.date).toLocaleDateString("en-US", {
                              day: "2-digit",
                              month: "short",
                              year: "numeric",
                            })}
                          </span>
                        </p>
                        <p className="flex justify-between">
                          <span className="text-slate-500">Cloud Coverage:</span>
                          <span className="font-semibold text-slate-200">
                            {result.after.cloud_cover}%
                          </span>
                        </p>
                        <p className="flex justify-between truncate" title={result.after.product_id}>
                          <span className="text-slate-500">Product ID:</span>
                          <span className="truncate max-w-[200px] text-slate-400">
                            {result.after.product_id}
                          </span>
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            )}

            {/* DETECTED CHANGE PANEL */}
            {(viewMode === "CHANGE" || viewMode === "SIDE_BY_SIDE" || viewMode === "SPLIT") && (
              <section className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-extrabold text-purple-300 flex items-center">
                      <Activity className="w-5 h-5 text-purple-400 mr-2" />
                      DETECTED CHANGE LAYER
                    </h2>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Computed spectral difference: ABS(AFTER - BEFORE)
                    </p>
                  </div>
                  <span className="text-xs bg-purple-500/10 text-purple-400 px-3 py-1 rounded-xl border border-purple-500/30 font-mono">
                    Optical Shift Signal
                  </span>
                </div>

                <div className="aspect-video md:aspect-[21/9] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 relative">
                  <img
                    src={result.comparison.difference_image_url}
                    alt="Computed Optical Spectral Change Image"
                    className="w-full h-full object-cover"
                    style={{ transform: `scale(${zoomLevel})` }}
                  />
                </div>
              </section>
            )}

            {/* MAP & DAMAGE ASSESSMENT GRID */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* INTERACTIVE AOI MAP */}
              <div className="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                  <h3 className="text-lg font-bold text-slate-200 mb-3 flex items-center">
                    <MapPin className="w-5 h-5 text-sky-400 mr-2" />
                    Target AOI & Bounding Box Map
                  </h3>
                  <div
                    ref={mapContainerRef}
                    className="w-full h-64 bg-slate-950 rounded-xl overflow-hidden border border-slate-800"
                  />
                </div>
                <div className="mt-4 text-xs text-slate-400 flex items-center justify-between font-mono">
                  <span>
                    Lat: {result.location.latitude.toFixed(4)}, Lon:{" "}
                    {result.location.longitude.toFixed(4)}
                  </span>
                  <span>AOI Size: ~500m × 500m</span>
                </div>
              </div>

              {/* DAMAGE ASSESSMENT RESULT CARD */}
              <div className="lg:col-span-6 bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <h3 className="text-lg font-bold text-slate-200 flex items-center">
                      <Sparkles className="w-5 h-5 text-indigo-400 mr-2" />
                      Damage Assessment Result
                    </h3>
                    <span className="text-xs text-slate-400 font-mono">
                      Phase 9 Certified Model
                    </span>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                        Predicted Class
                      </span>
                      <span
                        className={`inline-block px-4 py-2 rounded-2xl text-2xl font-black border shadow-lg ${getBadgeStyle(
                          result.model.class
                        )}`}
                      >
                        {result.model.class}
                      </span>
                    </div>

                    <div>
                      <div className="flex items-center justify-between text-xs font-semibold text-slate-400 mb-1">
                        <span>Confidence Score</span>
                        <span className="text-slate-200 text-sm">
                          {(result.model.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${result.model.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-4 border-t border-slate-800 text-[11px] text-slate-400 space-y-1">
                  <p>
                    <span className="font-semibold text-slate-300">Model Checkpoint:</span>{" "}
                    data/tamil_nadu/final/checkpoints/tamil_nadu_phase9_best.pt
                  </p>
                  <p className="text-slate-500">
                    Model architecture: Dual ResNet34 Encoders with Feature Difference Fusion.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
