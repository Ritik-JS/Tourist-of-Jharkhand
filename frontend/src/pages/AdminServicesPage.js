import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { ArrowLeft, Search, Filter, Users, MapPin, IndianRupee, Star, Loader2, Phone, Eye, AlertTriangle, Trash2, X, ToggleLeft, ToggleRight } from 'lucide-react';
import { providersAPI, destinationsAPI, adminAPI } from '../services/api';
import { useToast } from '../hooks/use-toast';

const AdminServicesPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [services, setServices] = useState([]);
  const [destinations, setDestinations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [destinationFilter, setDestinationFilter] = useState('all');
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedService, setSelectedService] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    if (user?.role === 'admin') {
      fetchData();
    }
  }, [user]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [servicesData, destinationsData] = await Promise.all([
        providersAPI.getAll(),
        destinationsAPI.getAll()
      ]);
      
      setServices(servicesData);
      setDestinations(destinationsData);
    } catch (error) {
      console.error('Error fetching data:', error);
      toast({
        title: "Error",
        description: "Failed to load services data",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  if (!user || user.role !== 'admin') {
    navigate('/login');
    return null;
  }

  // Filter services based on search and filters
  const filteredServices = services.filter(service => {
    const matchesSearch = service.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         service.service_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         service.category.toLowerCase().includes(searchTerm.toLowerCase());
    
    const matchesStatus = statusFilter === 'all' || 
                         (statusFilter === 'active' && service.is_active) ||
                         (statusFilter === 'inactive' && !service.is_active);
    
    const matchesCategory = categoryFilter === 'all' || service.category === categoryFilter;
    
    const matchesDestination = destinationFilter === 'all' || service.destination_id === destinationFilter;

    return matchesSearch && matchesStatus && matchesCategory && matchesDestination;
  });

  const categories = [...new Set(services.map(service => service.category))];
  const activeServices = services.filter(s => s.is_active).length;
  const inactiveServices = services.filter(s => !s.is_active).length;

  const handleDeleteService = async () => {
    if (!selectedService || actionLoading) return;

    try {
      setActionLoading(true);
      await adminAPI.deleteProvider(selectedService.id);
      toast({
        title: "Success",
        description: "Service deleted successfully",
      });
      await fetchData();
      setShowDeleteModal(false);
      setSelectedService(null);
    } catch (error) {
      console.error('Error deleting service:', error);
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to delete service",
        variant: "destructive",
      });
    } finally {
      setActionLoading(false);
    }
  };

  const handleToggleStatus = async (service) => {
    if (actionLoading) return;

    try {
      setActionLoading(true);
      await adminAPI.toggleProviderStatus(service.id);
      toast({
        title: "Success",
        description: `Service ${service.is_active ? 'deactivated' : 'activated'} successfully`,
      });
      await fetchData();
    } catch (error) {
      console.error('Error toggling service status:', error);
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to update service status",
        variant: "destructive",
      });
    } finally {
      setActionLoading(false);
    }
  };

  const openDeleteModal = (service) => {
    setSelectedService(service);
    setShowDeleteModal(true);
  };

  const closeDeleteModal = () => {
    setShowDeleteModal(false);
    setSelectedService(null);
  };

  const renderStars = (rating) => {
    const numRating = parseFloat(rating) || 0;
    return (
      <div className="flex items-center">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`h-4 w-4 ${
              star <= numRating ? 'text-yellow-400 fill-current' : 'text-gray-300'
            }`}
          />
        ))}
        <span className="ml-1 text-sm text-gray-600">({numRating.toFixed(1)})</span>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link to="/admin-dashboard">
                <Button variant="outline" size="sm" className="hover:bg-gray-50">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Dashboard
                </Button>
              </Link>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Manage Services</h1>
                <p className="text-gray-600 mt-1">Manage all tourism service providers and their status</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-blue-100">Total Services</p>
                  <p className="text-3xl font-bold">{services.length}</p>
                </div>
                <Users className="h-8 w-8 text-blue-200" />
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-green-100">Active Services</p>
                  <p className="text-3xl font-bold">{activeServices}</p>
                </div>
                <div className="h-8 w-8 bg-green-400 rounded-full flex items-center justify-center">
                  <div className="h-4 w-4 bg-white rounded-full"></div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-red-500 to-red-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-red-100">Inactive Services</p>
                  <p className="text-3xl font-bold">{inactiveServices}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-red-200" />
              </div>
            </CardContent>
          </Card>
          
          <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-purple-100">Categories</p>
                  <p className="text-3xl font-bold">{categories.length}</p>
                </div>
                <Filter className="h-8 w-8 text-purple-200" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <Card className="mb-6 shadow-lg">
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Search Services
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search by name, service, or category..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Status Filter
                </label>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Status</option>
                  <option value="active">Active Only</option>
                  <option value="inactive">Inactive Only</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Category Filter
                </label>
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Categories</option>
                  {categories.map(category => (
                    <option key={category} value={category}>
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Destination Filter
                </label>
                <select
                  value={destinationFilter}
                  onChange={(e) => setDestinationFilter(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="all">All Destinations</option>
                  {destinations.map(dest => (
                    <option key={dest.id} value={dest.id}>
                      {dest.location}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Services List */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Loading services...</p>
            </div>
          </div>
        ) : filteredServices.length === 0 ? (
          <div className="text-center py-20">
            <Users className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Services Found</h3>
            <p className="text-gray-600 mb-4">
              {searchTerm || statusFilter !== 'all' || categoryFilter !== 'all' || destinationFilter !== 'all' 
                ? 'No services match your current filters.' 
                : 'No services available in the system.'}
            </p>
          </div>
        ) : (
          <div className="space-y-6">
            {filteredServices.map((service) => (
              <Card key={service.id} className="overflow-hidden">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-purple-50">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <CardTitle className="text-xl text-gray-900">
                          {service.name}
                        </CardTitle>
                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                          service.is_active 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {service.is_active ? 'Active' : 'Inactive'}
                        </span>
                      </div>
                      <p className="text-lg font-medium text-blue-600 mb-3">
                        {service.service_name}
                      </p>
                      <div className="flex items-center space-x-6 text-sm text-gray-600">
                        <span className="inline-flex items-center px-2 py-1 bg-purple-100 text-purple-800 rounded-full">
                          {service.category}
                        </span>
                        <div className="flex items-center">
                          <MapPin className="h-4 w-4 mr-1" />
                          {service.location || 'Location not set'}
                        </div>
                        <div className="flex items-center">
                          <IndianRupee className="h-4 w-4 mr-1" />
                          ₹{service.price}
                        </div>
                        <div className="flex items-center">
                          <Phone className="h-4 w-4 mr-1" />
                          {service.contact}
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-col items-end space-y-2">
                      {renderStars(service.rating)}
                      <p className="text-sm text-gray-500">
                        Provider ID: {service.id}
                      </p>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="p-6">
                  <div className="mb-4">
                    <h4 className="font-medium text-gray-900 mb-2">Service Description</h4>
                    <p className="text-gray-600">{service.description}</p>
                  </div>
                  
                  {service.destination_name && (
                    <div className="mb-4">
                      <h4 className="font-medium text-gray-900 mb-2">Associated Destination</h4>
                      <p className="text-gray-600">{service.destination_name}</p>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <div className="text-sm text-gray-500">
                      <p>User ID: {service.user_id}</p>
                      <p>Created: {service.created_at ? new Date(service.created_at).toLocaleDateString() : 'N/A'}</p>
                    </div>
                    <div className="flex space-x-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleToggleStatus(service)}
                        disabled={actionLoading}
                        className={`${service.is_active ? 'hover:bg-orange-50' : 'hover:bg-green-50'}`}
                      >
                        {service.is_active ? (
                          <ToggleRight className="h-4 w-4 text-orange-600" />
                        ) : (
                          <ToggleLeft className="h-4 w-4 text-green-600" />
                        )}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => openDeleteModal(service)}
                        disabled={actionLoading}
                        className="hover:bg-red-50 text-red-600"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full">
            <div className="flex items-center justify-between p-6 border-b">
              <h2 className="text-xl font-bold text-gray-900">Delete Service</h2>
              <Button variant="ghost" size="sm" onClick={closeDeleteModal}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="p-6 text-center">
              <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Are you sure?</h3>
              <p className="text-gray-600 mb-6">
                This will permanently delete "{selectedService?.name}" service. This action cannot be undone.
              </p>
              <div className="flex justify-center space-x-4">
                <Button variant="outline" onClick={closeDeleteModal}>
                  Cancel
                </Button>
                <Button 
                  variant="destructive" 
                  onClick={handleDeleteService}
                  disabled={actionLoading}
                >
                  {actionLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                  Delete Service
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminServicesPage;