import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTranslation } from '../hooks/useTranslation';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { MapPin, Eye, EyeOff } from 'lucide-react';
import { useToast } from '../hooks/use-toast';
import LanguageToggle from '../components/LanguageToggle';

const LoginPage = () => {
  const [loginData, setLoginData] = useState({ email: '', password: '' });
  const [registerData, setRegisterData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    phone: '',
    role: 'tourist'
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { t } = useTranslation();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const user = await login(loginData.email, loginData.password);
      toast({
        title: t('loginSuccessful'),
        description: `${t('welcomeBack')}, ${user.name}!`,
      });
      
      // Redirect based on user role
      switch (user.role) {
        case 'admin':
          navigate('/admin');
          break;
        case 'provider':
          navigate('/provider-dashboard');
          break;
        case 'tourist':
          navigate('/tourist-dashboard');
          break;
        default:
          navigate('/');
      }
    } catch (error) {
      toast({
        title: t('loginFailed'),
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    if (registerData.password !== registerData.confirmPassword) {
      toast({
        title: t('registrationFailed'),
        description: t('passwordsDoNotMatch'),
        variant: "destructive",
      });
      setLoading(false);
      return;
    }

    try {
      const { confirmPassword, ...userData } = registerData;
      const user = await register(userData);
      toast({
        title: t('registrationSuccessful'),
        description: `${t('welcomeToJharkhandTourism')}, ${user.name}!`,
      });
      
      // Redirect based on user role
      switch (user.role) {
        case 'admin':
          navigate('/admin');
          break;
        case 'provider':
          navigate('/provider-dashboard');
          break;
        case 'tourist':
          navigate('/tourist-dashboard');
          break;
        default:
          navigate('/');
      }
    } catch (error) {
      toast({
        title: t('registrationFailed'),
        description: error.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-green-100 flex items-center justify-center p-4">
      {/* Language Toggle - Fixed Position */}
      <div className="fixed top-4 right-4 z-10">
        <LanguageToggle />
      </div>

      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <Link to="/" className="flex items-center justify-center space-x-2 mb-4">
            <MapPin className="h-8 w-8 text-green-600" />
            <span className="text-2xl font-bold text-foreground">
              {t('exploreJharkhand')}
            </span>
          </Link>
          <p className="text-muted-foreground">
            {t('welcomeToJharkhand')}
          </p>
        </div>

        <Card className="shadow-xl">
          <CardHeader>
            <CardTitle className="text-center">{t('welcome')}</CardTitle>
            <CardDescription className="text-center">
              {t('signInCreateAccount')}
            </CardDescription>
          </CardHeader>
          
          <CardContent>
            <Tabs defaultValue="login" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="login">{t('login')}</TabsTrigger>
                <TabsTrigger value="register">{t('register')}</TabsTrigger>
              </TabsList>
              
              {/* Login Tab */}
              <TabsContent value="login" className="space-y-4">
                <form onSubmit={handleLogin} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="email">{t('email')}</Label>
                    <Input
                      id="email"
                      type="email"
                      placeholder={t('enterEmail')}
                      value={loginData.email}
                      onChange={(e) => setLoginData({...loginData, email: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="password">{t('password')}</Label>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        placeholder={t('enterPassword')}
                        value={loginData.password}
                        onChange={(e) => setLoginData({...loginData, password: e.target.value})}
                        required
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                  
                  
                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? t('signingIn') : t('signIn')}
                  </Button>
                </form>
              </TabsContent>
              
              {/* Register Tab */}
              <TabsContent value="register" className="space-y-4">
                <form onSubmit={handleRegister} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">{t('fullName')}</Label>
                    <Input
                      id="name"
                      type="text"
                      placeholder={t('enterFullName')}
                      value={registerData.name}
                      onChange={(e) => setRegisterData({...registerData, name: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="reg-email">{t('email')}</Label>
                    <Input
                      id="reg-email"
                      type="email"
                      placeholder={t('enterEmail')}
                      value={registerData.email}
                      onChange={(e) => setRegisterData({...registerData, email: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="phone">{t('phoneNumber')}</Label>
                    <Input
                      id="phone"
                      type="tel"
                      placeholder={t('enterPhoneNumber')}
                      value={registerData.phone}
                      onChange={(e) => setRegisterData({...registerData, phone: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="role">{t('role')}</Label>
                    <Select onValueChange={(value) => setRegisterData({...registerData, role: value})}>
                      <SelectTrigger>
                        <SelectValue placeholder={t('selectRole')} />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tourist">{t('tourist')}</SelectItem>
                        <SelectItem value="provider">{t('serviceProvider')}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="reg-password">{t('password')}</Label>
                    <Input
                      id="reg-password"
                      type="password"
                      placeholder={t('createPassword')}
                      value={registerData.password}
                      onChange={(e) => setRegisterData({...registerData, password: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="confirm-password">{t('confirmPassword')}</Label>
                    <Input
                      id="confirm-password"
                      type="password"
                      placeholder={t('confirmYourPassword')}
                      value={registerData.confirmPassword}
                      onChange={(e) => setRegisterData({...registerData, confirmPassword: e.target.value})}
                      required
                    />
                  </div>
                  
                  <Button type="submit" className="w-full" disabled={loading}>
                    {loading ? t('creatingAccount') : t('createAccount')}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        
        <div className="text-center mt-6">
          <Link to="/" className="text-muted-foreground hover:text-foreground transition-colors">
            {t('backToHomepage')}
          </Link>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;