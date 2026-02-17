import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Star, MapPin, IndianRupee, Calendar, Users, X, Heart } from 'lucide-react';
import { wishlistAPI } from '../services/api';
import { useToast } from '../hooks/use-toast';
import ProviderSelectionModal from './ProviderSelectionModal';



const DestinationModal = ({ destination, isOpen, onClose }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [isAddingToWishlist, setIsAddingToWishlist] = useState(false);
  const [isInWishlist, setIsInWishlist] = useState(false);
  const [showProviderModal, setShowProviderModal] = useState(false);
  
  if (!destination) return null;

  const handleSelectProvider = () => {
    if (!user) {
      toast({
        title: "Authentication Required",
        description: "Please login to select service providers",
        variant: "destructive",
      });
      navigate('/login');
      return;
    }
    setShowProviderModal(true);
  };

  const handleAddToWishlist = async () => {
    if (!user) {
      toast({
        title: "Authentication Required",
        description: "Please login to add destinations to your wishlist",
        variant: "destructive",
      });
      return;
    }

    if (user.role !== 'tourist') {
      toast({
        title: "Access Denied",
        description: "Only tourists can add destinations to wishlist",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsAddingToWishlist(true);
      await wishlistAPI.add(destination.id);
      setIsInWishlist(true);
      toast({
        title: "Success",
        description: "Destination added to your wishlist!",
      });
    } catch (error) {
      console.error('Error adding to wishlist:', error);
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to add to wishlist",
        variant: "destructive",
      });
    } finally {
      setIsAddingToWishlist(false);
    }
  };

  const handleRemoveFromWishlist = async () => {
    try {
      setIsAddingToWishlist(true);
      await wishlistAPI.remove(destination.id);
      setIsInWishlist(false);
      toast({
        title: "Success",
        description: "Destination removed from your wishlist",
      });
    } catch (error) {
      console.error('Error removing from wishlist:', error);
      toast({
        title: "Error",
        description: "Failed to remove from wishlist",
        variant: "destructive",
      });
    } finally {
      setIsAddingToWishlist(false);
    }
  };

  return (
    <>
      <Dialog open={isOpen} onOpenChange={onClose}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              <span className="text-2xl font-bold">{destination.name}</span>
              <Button variant="ghost" size="sm" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Image Section */}
            <div className="space-y-4">
              <div className="relative">
                <img
                  src={destination.image_url}
                  alt={destination.name}
                  className="w-full h-64 lg:h-80 object-cover rounded-lg"
                />
                <div className="absolute top-4 left-4">
                  <Badge className="bg-green-600 text-white">
                    {destination.category}
                  </Badge>
                </div>
                <div className="absolute top-4 right-4 flex items-center space-x-1 bg-black/60 backdrop-blur-sm rounded-full px-3 py-1">
                  <Star className="h-4 w-4 text-yellow-400 fill-current" />
                  <span className="text-white text-sm font-medium">{destination.rating}</span>
                </div>
              </div>
              
              {/* YouTube Video Player */}
              {destination.videoUrl && (
                <div className="relative aspect-video rounded-lg overflow-hidden">
                  <iframe
                    src={destination.videoUrl}
                    title={`${destination.name} Video`}
                    className="w-full h-full"
                    frameBorder="0"
                    allowFullScreen
                  />
                </div>
              )}
            </div>
            
            {/* Details Section */}
            <div className="space-y-6">
              <div>
                <div className="flex items-center text-muted-foreground mb-3">
                  <MapPin className="h-5 w-5 mr-2" />
                  <span>{destination.location}</span>
                </div>
                
                <p className="text-muted-foreground leading-relaxed">
                  {destination.description}
                </p>
              </div>
              
              {/* Highlights */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Highlights</h3>
                <div className="grid grid-cols-2 gap-2">
                  {destination.highlights.map((highlight, index) => (
                    <div key={index} className="flex items-center p-2 bg-green-50 rounded-lg">
                      <div className="w-2 h-2 bg-green-600 rounded-full mr-3"></div>
                      <span className="text-sm font-medium">{highlight}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Pricing and Booking */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="flex items-center text-green-600 font-medium text-2xl">
                      
                      <span>Contact Your Service Provider For Pricing</span>
                    </div>
                    <p className="text-sm text-muted-foreground">based on your service provider</p>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Star className="h-5 w-5 text-yellow-400 fill-current" />
                    <span className="font-semibold">{destination.rating}</span>
                    <span className="text-muted-foreground text-sm">(124 reviews)</span>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="flex items-center p-3 border rounded-lg">
                    <Calendar className="h-5 w-5 text-green-600 mr-2" />
                    <div>
                      <p className="text-sm font-medium">Duration</p>
                      <p className="text-xs text-muted-foreground">2-3 days</p>
                    </div>
                  </div>
                  <div className="flex items-center p-3 border rounded-lg">
                    <Users className="h-5 w-5 text-green-600 mr-2" />
                    <div>
                      <p className="text-sm font-medium">Group Size</p>
                      <p className="text-xs text-muted-foreground">2-8 people</p>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-3">
                  <Button 
                    onClick={handleSelectProvider}
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                  >
                    Select Service Provider
                  </Button>
                  {user && user.role === 'tourist' && (
                    <Button 
                      variant="outline" 
                      className="w-full"
                      onClick={isInWishlist ? handleRemoveFromWishlist : handleAddToWishlist}
                      disabled={isAddingToWishlist}
                    >
                      {isAddingToWishlist ? (
                        <div className="flex items-center">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-green-600 mr-2"></div>
                          {isInWishlist ? 'Removing...' : 'Adding...'}
                        </div>
                      ) : (
                        <div className="flex items-center">
                          <Heart className={`h-4 w-4 mr-2 ${isInWishlist ? 'fill-red-500 text-red-500' : ''}`} />
                          {isInWishlist ? 'Remove from Wishlist' : 'Add to Wishlist'}
                        </div>
                      )}
                    </Button>
                  )}
                </div>
              </div>
              
              {/* Additional Info */}
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Best Time to Visit:</span>
                  <span className="font-medium">October - March</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Difficulty Level:</span>
                  <span className="font-medium">Easy to Moderate</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Languages:</span>
                  <span className="font-medium">Hindi, English, Local</span>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Provider Selection Modal */}
      <ProviderSelectionModal
        destination={destination}
        isOpen={showProviderModal}
        onClose={() => setShowProviderModal(false)}
      />
    </>
  );
};

export default DestinationModal;