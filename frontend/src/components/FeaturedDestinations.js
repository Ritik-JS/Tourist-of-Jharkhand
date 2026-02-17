import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { destinationsAPI } from '../services/api';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Star, MapPin, IndianRupee, Loader2 } from 'lucide-react';
import DestinationModal from './DestinationModal';
import { useToast } from '../hooks/use-toast';

const FeaturedDestinations = () => {
  const [selectedDestination, setSelectedDestination] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [destinations, setDestinations] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const { toast } = useToast();

  useEffect(() => {
    fetchDestinations();
  }, []);

  const fetchDestinations = async () => {
    try {
      setLoading(true);
      const data = await destinationsAPI.getAll();
      setDestinations(data);
    } catch (error) {
      console.error('Error fetching destinations:', error);
      toast({
        title: "Error",
        description: "Failed to load destinations",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const featuredDestinations = destinations.slice(0, 6);

  const handleLearnMore = (destination) => {
    setSelectedDestination(destination);
    setIsModalOpen(true);
  };

  const handleViewAllDestinations = () => {
    navigate('/destinations');
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedDestination(null);
  };

  return (
    <section className="py-20 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">
            Plan your trip to Jharkhand
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Are you visiting Jharkhand soon? Below you'll find a selection of seasonal highlights 
            from the best things to do to hidden gems and more!
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Loading featured destinations...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {featuredDestinations.map((destination) => (
                <Card 
                  key={destination.id}
                  className="group overflow-hidden border-0 shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2 bg-card cursor-pointer"
                  onClick={() => handleLearnMore(destination)}
                >
                  <div className="relative overflow-hidden">
                    <img
                      src={destination.image_url}
                      alt={destination.name}
                      className="w-full h-64 object-cover transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute top-4 right-4">
                      <Badge className="bg-green-600 text-white">
                        {destination.category}
                      </Badge>
                    </div>
                    <div className="absolute top-4 left-4 flex items-center space-x-1 bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
                      <Star className="h-4 w-4 text-yellow-400 fill-current" />
                      <span className="text-white text-sm font-medium">{destination.rating}</span>
                    </div>
                  </div>
                  
                  <CardContent className="p-6">
                    <h3 className="text-xl font-bold mb-2 group-hover:text-green-600 transition-colors">
                      {destination.name}
                    </h3>
                    
                    <div className="flex items-center text-muted-foreground mb-3">
                      <MapPin className="h-4 w-4 mr-1" />
                      <span className="text-sm">{destination.location}</span>
                    </div>
                    
                    <p className="text-muted-foreground mb-4 line-clamp-2">
                      {destination.description}
                    </p>
                    
                    <div className="flex flex-wrap gap-2 mb-4">
                      {destination.highlights.slice(0, 3).map((highlight, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {highlight}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center text-green-600 font-medium">
                        
                        <span>Contact Your Service Provider For Pricing</span>
                      </div>
                      <button 
                        className="text-green-600 hover:text-green-700 font-medium text-sm transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleLearnMore(destination);
                        }}
                      >
                        Learn more →
                      </button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <div className="text-center mt-12">
              <Button 
                onClick={handleViewAllDestinations}
                className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 text-lg font-semibold transition-colors"
              >
                View All Destinations
              </Button>
            </div>
          </>
        )}
      </div>

      {/* Destination Modal */}
      <DestinationModal
        destination={selectedDestination}
        isOpen={isModalOpen}
        onClose={handleCloseModal}
      />
    </section>
  );
};

export default FeaturedDestinations;