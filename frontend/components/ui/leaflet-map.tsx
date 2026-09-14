'use client';

import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface GeoJSONFeature {
  type: 'Feature';
  id?: string | number;
  properties: {
    building_id?: string;
    osm_id?: number | string;
    prediction?: string;
    damage_prediction?: string;
    confidence?: number;
    [key: string]: any;
  };
  geometry: {
    type: 'Polygon' | 'Point';
    coordinates: any;
  };
}

interface GeoJSONData {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}

interface LeafletMapProps {
  data: GeoJSONData | null;
  selectedFeature: GeoJSONFeature | null;
  onSelectBuilding: (feature: GeoJSONFeature) => void;
  filterStatus?: string;
  searchQueryLocation?: { lat: number; lon: number } | null;
}

// Configure default Leaflet icon paths
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

export default function LeafletMap({ data, selectedFeature, onSelectBuilding, filterStatus, searchQueryLocation }: LeafletMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const geojsonLayerRef = useRef<L.GeoJSON | null>(null);
  const highlightLayerRef = useRef<L.GeoJSON | null>(null);
  const searchMarkerRef = useRef<L.Marker | null>(null);
  const connectorLineRef = useRef<L.Polyline | null>(null);

  // Initialize Leaflet Map Instance
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapRef.current) return;

    console.log('[LEAFLET] Initializing Leaflet map with OpenStreetMap tiles...');

    const map = L.map(mapContainerRef.current, {
      center: [13.0429, 80.2064],
      zoom: 14,
      zoomControl: true,
    });

    // Standard OpenStreetMap Tile Layer (100% Token-Free)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    mapRef.current = map;

    const resizeObserver = new ResizeObserver(() => {
      if (mapRef.current) {
        mapRef.current.invalidateSize();
      }
    });
    resizeObserver.observe(mapContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Sync GeoJSON Building Layer & Color Classification
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (geojsonLayerRef.current) {
      map.removeLayer(geojsonLayerRef.current);
      geojsonLayerRef.current = null;
    }

    if (!data || !data.features || data.features.length === 0) {
      console.log('[LEAFLET] No building features to display.');
      return;
    }

    console.log(`[LEAFLET] Rendering ${data.features.length} building features on Leaflet map...`);

    const filteredFeatures = filterStatus && filterStatus !== 'ALL'
      ? data.features.filter(f => (f.properties.prediction || f.properties.damage_prediction) === filterStatus)
      : data.features;

    const geojsonLayer = L.geoJSON({ type: 'FeatureCollection', features: filteredFeatures } as any, {
      style: (feature) => {
        const pred = feature?.properties?.prediction || feature?.properties?.damage_prediction;
        let color = '#94a3b8'; // Neutral Gray
        if (pred === 'INTACT') color = '#10b981'; // Green
        else if (pred === 'DAMAGED') color = '#f59e0b'; // Amber/Orange
        else if (pred === 'DESTROYED') color = '#ef4444'; // Red

        return {
          fillColor: color,
          fillOpacity: 0.65,
          color: '#ffffff',
          weight: 1.2,
          opacity: 0.9,
        };
      },
      onEachFeature: (feature, layer) => {
        layer.on('click', () => {
          onSelectBuilding(feature as unknown as GeoJSONFeature);
        });

        const props = feature.properties;
        const bldId = props.building_id || props.osm_id || 'Building';
        const pred = props.prediction || props.damage_prediction || 'UNKNOWN';
        const conf = props.confidence ? `${(props.confidence * 100).toFixed(1)}%` : 'N/A';

        layer.bindTooltip(
          `<div class="font-sans text-xs font-semibold px-1 py-0.5">
            <div class="text-cyan-400 font-bold">${bldId}</div>
            <div class="text-slate-200 font-normal mt-0.5">Status: <span class="font-bold">${pred}</span> (${conf})</div>
          </div>`,
          { sticky: true, className: 'leaflet-custom-tooltip' }
        );
      },
    }).addTo(map);

    geojsonLayerRef.current = geojsonLayer;

    // Auto-fit Leaflet Bounds to Loaded Features
    if (filteredFeatures.length > 0) {
      const bounds = geojsonLayer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 16, animate: true });
      }
    }
  }, [data, filterStatus]);

  // Sync Search Location Marker Pin & Connector Line
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (searchMarkerRef.current) {
      map.removeLayer(searchMarkerRef.current);
      searchMarkerRef.current = null;
    }
    if (connectorLineRef.current) {
      map.removeLayer(connectorLineRef.current);
      connectorLineRef.current = null;
    }

    if (!searchQueryLocation) return;

    const searchIcon = L.divIcon({
      className: 'custom-search-pin',
      html: `<div style="
        width: 24px;
        height: 24px;
        background: #0284c7;
        border: 2px solid #ffffff;
        border-radius: 50%;
        box-shadow: 0 0 12px #38bdf8;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
      ">🔍</div>`,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    });

    const searchMarker = L.marker([searchQueryLocation.lat, searchQueryLocation.lon], { icon: searchIcon })
      .bindTooltip('Searched Coordinates', { permanent: true, direction: 'top', className: 'text-xs bg-slate-900 text-sky-400 font-bold border border-sky-500' })
      .addTo(map);

    searchMarkerRef.current = searchMarker;

    // Draw dashed polyline connector if a building is selected
    if (selectedFeature && selectedFeature.geometry) {
      let bldLat = searchQueryLocation.lat;
      let bldLon = searchQueryLocation.lon;
      const geom = selectedFeature.geometry;

      if (geom.type === 'Point' && Array.isArray(geom.coordinates)) {
        bldLon = geom.coordinates[0];
        bldLat = geom.coordinates[1];
      } else if (geom.type === 'Polygon' && Array.isArray(geom.coordinates)) {
        const ring = geom.coordinates[0];
        let sumLat = 0, sumLon = 0;
        ring.forEach((c: number[]) => { sumLon += c[0]; sumLat += c[1]; });
        bldLat = sumLat / ring.length;
        bldLon = sumLon / ring.length;
      }

      const connector = L.polyline([
        [searchQueryLocation.lat, searchQueryLocation.lon],
        [bldLat, bldLon]
      ], {
        color: '#38bdf8',
        weight: 2.5,
        dashArray: '6, 6',
        opacity: 0.85
      }).addTo(map);

      connectorLineRef.current = connector;
    }
  }, [searchQueryLocation, selectedFeature]);

  // Sync Highlight Layer & Centralized Building Focus
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (highlightLayerRef.current) {
      map.removeLayer(highlightLayerRef.current);
      highlightLayerRef.current = null;
    }

    if (!selectedFeature) return;

    const highlightLayer = L.geoJSON(selectedFeature as any, {
      style: {
        fillColor: '#38bdf8',
        fillOpacity: 0.55,
        color: '#ffffff',
        weight: 3.5,
        opacity: 1.0,
      },
    }).addTo(map);

    highlightLayerRef.current = highlightLayer;

    const bounds = highlightLayer.getBounds();
    if (bounds.isValid()) {
      map.flyToBounds(bounds, { maxZoom: 18, duration: 1.0 });
    }
  }, [selectedFeature]);

  return (
    <div className="relative w-full h-full min-h-[500px] overflow-hidden bg-slate-950">
      <div
        ref={mapContainerRef}
        className="absolute inset-0 w-full h-full min-h-[500px]"
        style={{ width: '100%', height: '100%', minHeight: '500px' }}
      />
    </div>
  );
}
