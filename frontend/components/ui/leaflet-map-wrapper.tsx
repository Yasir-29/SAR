"use client";

import React, { useEffect, useRef } from "react";

interface LeafletMapWrapperProps {
  lat: number;
  lon: number;
  zoom?: number;
}

export default function LeafletMapWrapper({ lat, lon, zoom = 15 }: LeafletMapWrapperProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletInstanceRef = useRef<any>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const container = mapRef.current;
    if (!container) return;

    import("leaflet").then((L) => {
      // Fix default Leaflet icon paths
      delete (L.Icon.Default.prototype as any)._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
        iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
        shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
      });

      if (leafletInstanceRef.current) {
        leafletInstanceRef.current.remove();
      }

      const map = L.map(container).setView([lat, lon], zoom);
      leafletInstanceRef.current = map;

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors | Copernicus CDSE",
        maxZoom: 19,
      }).addTo(map);

      // Target Coordinate Marker
      L.marker([lat, lon])
        .addTo(map)
        .bindPopup(`<b>Disaster Coordinate</b><br/>Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`)
        .openPopup();

      // AOI Bounding Box (~500m x 500m)
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
      if (leafletInstanceRef.current) {
        leafletInstanceRef.current.remove();
        leafletInstanceRef.current = null;
      }
    };
  }, [lat, lon, zoom]);

  return (
    <div
      ref={mapRef}
      className="w-full h-64 bg-slate-950 rounded-2xl overflow-hidden border border-slate-800"
    />
  );
}
