import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import { Card, CardContent } from './ui/card';
import { Button } from './ui/button';
import { MapPin, Star, Layers, Satellite, Map as MapIcon } from 'lucide-react';
import ARVRMapLauncher from './ARVRMapLauncher';

// Enhanced Leaflet Map with multiple tile layers
const AlternativeMapView = ({ 
  destinations = [], 
  selectedDestination, 
  onDestinationSelect,
  center = [23.6102, 85.2799],
  zoom = 8,
  bounds = [[21.9, 83.3], [25.3, 87.5]]
}) => {
  const [mapStyle, setMapStyle] = useState('street'); // 'street', 'satellite', 'terrain'

  // Tile layer configurations
  const tileConfigs = {
    street: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "© OpenStreetMap contributors"
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "© Esri"
    },
    terrain: {
      url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
      attribution: "© OpenTopoMap contributors"
    }
  };

  // Custom marker icons
  const createCustomIcon = (category, isSelected = false) => {
    const colors = {
      'city': '#3B82F6',
      'nature': '#10B981',
      'wildlife': '#F59E0B',
      'religious': '#8B5CF6',
      'adventure': '#EF4444',
      'default': '#6B7280'
    };
    
    const color = colors[category?.toLowerCase()] || colors.default;
    const size = isSelected ? 32 : 24;
    const borderWidth = isSelected ? 4 : 3;
    
    return L.divIcon({
      html: `
        <div style="
          background-color: ${color}; 
          width: ${size}px; 
          height: ${size}px; 
          border-radius: 50%; 
          border: ${borderWidth}px solid white; 
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: bold;
          font-size: ${size/2}px;
        ">
          ${isSelected ? '📍' : '•'}
        </div>
      `,
      className: 'custom-marker',
      iconSize: [size, size],
      iconAnchor: [size/2, size/2],
    });
  };

  const handleDestinationClick = (destination) => {
    onDestinationSelect && onDestinationSelect(destination);
  };

  return (
    <div className="relative w-full h-full">
      {/* Map Style Controls */}
      <div className="absolute top-4 left-4 z-[1000] bg-white rounded-lg shadow-lg p-2">
        <div className="flex space-x-1">
          <Button
            onClick={() => setMapStyle('street')}
            size="sm"
            variant={mapStyle === 'street' ? 'default' : 'outline'}
            className="px-2 py-1"
          >
            <MapIcon className="h-3 w-3 mr-1" />
            Street
          </Button>
          <Button
            onClick={() => setMapStyle('satellite')}
            size="sm"
            variant={mapStyle === 'satellite' ? 'default' : 'outline'}
            className="px-2 py-1"
          >
            <Satellite className="h-3 w-3 mr-1" />
            Satellite
          </Button>
          <Button
            onClick={() => setMapStyle('terrain')}
            size="sm"
            variant={mapStyle === 'terrain' ? 'default' : 'outline'}
            className="px-2 py-1"
          >
            <Layers className="h-3 w-3 mr-1" />
            Terrain
          </Button>
        </div>
      </div>

      {/* AR/VR Controls */}
      <div className="absolute top-4 right-4 z-[1000]">
        <div className="bg-white rounded-lg shadow-lg p-3">
          <h4 className="text-sm font-semibold mb-2 text-gray-700">
            Immersive Experiences
          </h4>
          <ARVRMapLauncher
            destinations={destinations}
            selectedDestination={selectedDestination}
            onDestinationSelect={onDestinationSelect}
            layout="vertical"
            size="sm"
          />
        </div>
      </div>

      {/* Selected Destination Info */}
      {selectedDestination && (
        <div className="absolute bottom-4 left-4 z-[1000] max-w-sm">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-start space-x-3">
                {selectedDestination.image_url && (
                  <img 
                    src={selectedDestination.image_url} 
                    alt={selectedDestination.name}
                    className="w-16 h-16 object-cover rounded"
                  />
                )}
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 mb-1">
                    {selectedDestination.name}
                  </h3>
                  <div className="flex items-center text-gray-600 mb-2">
                    <MapPin className="h-3 w-3 mr-1" />
                    <span className="text-sm">{selectedDestination.location}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Star className="h-3 w-3 text-yellow-400 fill-current mr-1" />
                      <span className="text-sm">{selectedDestination.rating}</span>
                    </div>
                    <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                      {selectedDestination.category}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="mt-3">
                <ARVRMapLauncher
                  destinations={[selectedDestination]}
                  selectedDestination={selectedDestination}
                  onDestinationSelect={onDestinationSelect}
                  layout="horizontal"
                  size="sm"
                />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Map Legend */}
      <div className="absolute bottom-4 right-4 z-[1000] bg-white rounded-lg shadow-lg p-3">
        <h4 className="text-sm font-semibold mb-2 text-gray-700">Map Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>City Attractions</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
            <span>Nature & Hills</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
            <span>Wildlife</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
            <span>Religious Sites</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
            <span>Adventure Sports</span>
          </div>
        </div>
      </div>

      {/* Leaflet Map */}
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        bounds={bounds}
        className="z-0"
      >
        <TileLayer
          url={tileConfigs[mapStyle].url}
          attribution={tileConfigs[mapStyle].attribution}
        />
        
        {destinations.map((destination) => (
          <Marker
            key={destination.id}
            position={destination.coordinates}
            icon={createCustomIcon(
              destination.category, 
              selectedDestination?.id === destination.id
            )}
            eventHandlers={{
              click: () => handleDestinationClick(destination),
            }}
          >
            <Popup>
              <div className="max-w-xs">
                {destination.image_url && (
                  <img
                    src={destination.image_url}
                    alt={destination.name}
                    className="w-full h-20 object-cover rounded mb-2"
                  />
                )}
                <h3 className="font-semibold text-gray-900 mb-1">
                  {destination.name}
                </h3>
                <div className="flex items-center text-gray-600 mb-2">
                  <MapPin className="h-3 w-3 mr-1" />
                  <span className="text-xs">{destination.location}</span>
                </div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center">
                    <Star className="h-3 w-3 text-yellow-400 fill-current mr-1" />
                    <span className="text-xs">{destination.rating}</span>
                  </div>
                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                    {destination.category}
                  </span>
                </div>
                <p className="text-xs text-gray-600 mb-3 line-clamp-2">
                  {destination.description}
                </p>
                
                {/* AR/VR Launch Buttons in Popup */}
                <ARVRMapLauncher
                  destinations={[destination]}
                  selectedDestination={destination}
                  onDestinationSelect={onDestinationSelect}
                  layout="horizontal"
                  size="sm"
                />
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default AlternativeMapView;