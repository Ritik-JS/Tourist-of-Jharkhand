import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { 
  ArrowLeft, 
  MapPin, 
  Star, 
  Heart, 
  Share2, 
  Calendar, 
  Users, 
  Camera,
  Clock,
  IndianRupee,
  Loader2
} from 'lucide-react';
import { destinationsAPI, wishlistAPI } from '../services/api';
import { useToast } from '../hooks/use-toast';

const DestinationDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { toast } = useToast();
  const [destination, setDestination] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isWishlisted, setIsWishlisted] = useState(false);
  const [wishlistLoading, setWishlistLoading] = useState(false);

  useEffect(() => {
    fetchDestination();
    if (user && user.role === 'tourist') {
      checkWishlistStatus();
    }
  }, [id, user]);

  const fetchDestination = async () => {
    try {
      setLoading(true);
      const data = await destinationsAPI.getById(id);
      
      // Parse highlights if it's a JSON string
      let highlights = [];
      if (data.highlights) {
        try {
          highlights = typeof data.highlights === 'string' ? JSON.parse(data.highlights) : data.highlights;
        } catch (e) {
          highlights = [];
        }
      }
      
      setDestination({
        ...data,
        highlights: highlights,
        // Add some additional details for the detail page
        gallery: [
          data.image_url,
          'https://images.unsplash.com/photo-1544735716-392fe2489ffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80',
          'https://images.unsplash.com/photo-1551632811-561732d1e306?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80'
        ],
        bestTimeToVisit: 'October to March',
        duration: '2-3 days',
        difficulty: 'Easy',
        activities: ['Sightseeing', 'Photography', 'Cultural Tours', 'Nature Walks'],
        facilities: ['Parking', 'Restrooms', 'Food Court', 'Guide Services'],
        nearbyAttractions: [
          { name: 'Ranchi Lake', distance: '5 km' },
          { name: 'Jagannath Temple', distance: '8 km' },
          { name: 'Rock Garden', distance: '3 km' }
        ]
      });
    } catch (error) {
      console.error('Error fetching destination:', error);
      toast({
        title: "Error",
        description: "Failed to load destination details",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const checkWishlistStatus = async () => {
    try {
      const data = await wishlistAPI.checkStatus(id);
      setIsWishlisted(data.is_wishlisted);
    } catch (error) {
      console.error('Error checking wishlist status:', error);
    }
  };

  const handleWishlistToggle = async () => {
    if (!user) {
      navigate('/login');
      return;
    }

    if (user.role !== 'tourist') {
      toast({
        title: "Access Denied",
        description: "Only tourists can manage wishlist",
        variant: "destructive",
      });
      return;
    }

    try {
      setWishlistLoading(true);
      
      if (isWishlisted) {
        await wishlistAPI.remove(id);
        setIsWishlisted(false);
        toast({
          title: "Success",
          description: "Removed from wishlist",
        });
      } else {
        await wishlistAPI.add(id);
        setIsWishlisted(true);
        toast({
          title: "Success",
          description: "Added to wishlist",
        });
      }
    } catch (error) {
      console.error('Error toggling wishlist:', error);
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to update wishlist",
        variant: "destructive",
      });
    } finally {
      setWishlistLoading(false);
    }
  };

  const handleBookNow = () => {
    if (!user) {
      navigate('/login');
      return;
    }
    // Navigate to booking page with destination info
    navigate('/booking', { 
      state: { 
        destination: destination,
        isDestinationBooking: true 
      }
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">Loading destination details...</p>
        </div>
      </div>
    );
  }

  if (!destination) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Destination Not Found</h2>
          <p className="text-gray-600 mb-6">The destination you're looking for doesn't exist.</p>
          <Link to="/destinations">
            <Button className="bg-green-600 hover:bg-green-700">
              Browse All Destinations
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{destination.name}</h1>
                <div className="flex items-center text-gray-600">
                  <MapPin className="h-4 w-4 mr-1" />
                  <span>{destination.location}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleWishlistToggle}
                disabled={wishlistLoading}
                className={isWishlisted ? 'text-red-600 border-red-300' : ''}
              >
                {wishlistLoading ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Heart className={`h-4 w-4 mr-2 ${isWishlisted ? 'fill-current' : ''}`} />
                )}
                {isWishlisted ? 'Saved' : 'Save'}
              </Button>
              <Button variant="outline" size="sm">
                <Share2 className="h-4 w-4 mr-2" />
                Share
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image Gallery */}
            <Card>
              <CardContent className="p-0">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <img
                    src={destination.gallery[0]}
                    alt={destination.name}
                    className="w-full h-64 md:h-80 object-cover rounded-l-lg"
                  />
                  <div className="grid grid-cols-1 gap-2">
                    {destination.gallery.slice(1, 3).map((image, index) => (
                      <img
                        key={index}
                        src={image}
                        alt={`${destination.name} ${index + 2}`}
                        className="w-full h-32 md:h-39 object-cover rounded-r-lg"
                      />
                    ))}
                  </div>
                </div>
                <div className="p-4">
                  <Button variant="outline" size="sm">
                    <Camera className="h-4 w-4 mr-2" />
                    View All Photos
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Description */}
            <Card>
              <CardHeader>
                <CardTitle>About This Place</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 leading-relaxed mb-4">
                  {destination.description}
                </p>
                
                {/* Highlights */}
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Highlights</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {destination.highlights.map((highlight, index) => (
                      <div key={index} className="flex items-center text-gray-600">
                        <Star className="h-4 w-4 text-green-600 mr-2 flex-shrink-0" />
                        <span className="text-sm">{highlight}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Activities */}
            <Card>
              <CardHeader>
                <CardTitle>Things to Do</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {destination.activities.map((activity, index) => (
                    <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2">
                        <Camera className="h-6 w-6 text-green-600" />
                      </div>
                      <span className="text-sm text-gray-700">{activity}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Nearby Attractions */}
            <Card>
              <CardHeader>
                <CardTitle>Nearby Attractions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {destination.nearbyAttractions.map((attraction, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center">
                        <MapPin className="h-4 w-4 text-gray-500 mr-2" />
                        <span className="text-gray-700">{attraction.name}</span>
                      </div>
                      <span className="text-sm text-gray-500">{attraction.distance}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Booking Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Book Your Visit</span>
                  <div className="flex items-center">
                    <Star className="h-4 w-4 text-yellow-400 fill-current mr-1" />
                    <span className="text-sm">{destination.rating}</span>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-center">
                  <div className="flex items-center justify-center text-2xl font-bold text-green-600 mb-2">
                    <span>Contact Your Service Provider For Pricing</span>
                  </div>
                  <p className="text-sm text-gray-600">based on service provider</p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center text-gray-600">
                    <Clock className="h-4 w-4 mr-2" />
                    <span className="text-sm">Duration: {destination.duration}</span>
                  </div>
                  <div className="flex items-center text-gray-600">
                    <Calendar className="h-4 w-4 mr-2" />
                    <span className="text-sm">Best time: {destination.bestTimeToVisit}</span>
                  </div>
                  <div className="flex items-center text-gray-600">
                    <Users className="h-4 w-4 mr-2" />
                    <span className="text-sm">Difficulty: {destination.difficulty}</span>
                  </div>
                </div>

                <Button 
                  onClick={handleBookNow}
                  className="w-full bg-green-600 hover:bg-green-700"
                >
                  Book Now
                </Button>
                
                <Button variant="outline" className="w-full">
                  Contact Tour Guide
                </Button>
              </CardContent>
            </Card>

            {/* Facilities */}
            <Card>
              <CardHeader>
                <CardTitle>Facilities Available</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {destination.facilities.map((facility, index) => (
                    <div key={index} className="flex items-center text-gray-600">
                      <div className="w-2 h-2 bg-green-600 rounded-full mr-3"></div>
                      <span className="text-sm">{facility}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DestinationDetailPage;