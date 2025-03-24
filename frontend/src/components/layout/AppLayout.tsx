import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  useTheme,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  ListSubheader,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search as SearchIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
  Security as SecurityIcon,
  Build as BuildIcon,
  FolderSpecial as ProjectIcon,
  SmartToy as AIIcon,
  Chat as ChatIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Speed as MonitoringIcon,
} from '@mui/icons-material';
import { ProjectSelector } from '../projects/ProjectSelector';
import { UserProvider, useUser } from '../../contexts/UserContext';
import type { User } from '../../api/proxyApi';

const DRAWER_WIDTH = 240;

// Menu structure with grouping
const menuGroups = [
  {
    header: "Main",
    items: [
      { text: 'Projects', icon: <ProjectIcon />, path: '/projects' },
      { text: 'Reconnaissance', icon: <SearchIcon />, path: '/recon' },
      { text: 'Proxy', icon: <NetworkIcon />, path: '/proxy' },
      { text: 'Vulnerabilities', icon: <SecurityIcon />, path: '/vulnerabilities' },
    ]
  },
  {
    header: "AI Features",
    items: [
      { text: 'AI Hub', icon: <AIIcon />, path: '/ai' },
      { text: 'Chat', icon: <ChatIcon />, path: '/ai/chat' },
      { text: 'Analytics', icon: <AnalyticsIcon />, path: '/ai/analytics' },
      { text: 'Monitoring', icon: <MonitoringIcon />, path: '/ai/monitoring' },
    ]
  },
  {
    header: "Configuration",
    items: [
      { text: 'Tools', icon: <BuildIcon />, path: '/tools' },
      { text: 'Health', icon: <MonitoringIcon />, path: '/health' },
      { text: 'AI Settings', icon: <SettingsIcon />, path: '/settings/ai' },
    ]
  }
];

interface AppLayoutProps {
  children: React.ReactNode;
}

function AppLayoutContent({ children }: AppLayoutProps) {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const { users, currentUser, setCurrentUser } = useUser();

  const handleUserMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const drawer = (
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap>
          Anarchy Copilot
        </Typography>
      </Toolbar>
      <List>
        {menuGroups.map((group, index) => (
          <React.Fragment key={group.header}>
            {index > 0 && <Divider />}
            <ListSubheader>{group.header}</ListSubheader>
            {group.items.map((item) => (
              <ListItem
                button
                key={item.text}
                onClick={() => navigate(item.path)}
                selected={location.pathname === item.path}
                sx={{
                  '&.Mui-selected': {
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    '&:hover': {
                      backgroundColor: 'rgba(0, 255, 0, 0.2)',
                    },
                  },
                }}
              >
                <ListItemIcon sx={{ color: 'primary.main' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </React.Fragment>
        ))}
      </List>
    </Box>
  );

  // Find current menu item
  const currentMenuItem = menuGroups
    .flatMap(group => group.items)
    .find(item => item.path === location.pathname);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
          ml: { sm: `${DRAWER_WIDTH}px` },
          backgroundColor: 'background.paper',
          borderBottom: '1px solid rgba(0, 255, 0, 0.1)',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Box display="flex" justifyContent="space-between" alignItems="center" width="100%">
            <Typography variant="h6" noWrap component="div">
              {currentMenuItem?.text || 'Dashboard'}
            </Typography>
            <Box display="flex" alignItems="center">
              <Box sx={{ borderRadius: 1, p: 1, mr: 2 }}>
                <ProjectSelector onCreateClick={() => navigate('/projects')} />
              </Box>
              <Avatar onClick={handleUserMenuClick} sx={{ cursor: 'pointer', ml: 2 }}>
                {currentUser?.username[0] || '?'}
              </Avatar>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleUserMenuClose}
              >
                {users.map((user: User) => (
                  <MenuItem key={user.id} onClick={() => {
                    setCurrentUser(user);
                    handleUserMenuClose();
                  }}>
                    {user.username}
                  </MenuItem>
                ))}
              </Menu>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: DRAWER_WIDTH }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
              backgroundColor: 'background.paper',
              borderRight: '1px solid rgba(0, 255, 0, 0.1)',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${DRAWER_WIDTH}px)` },
          backgroundColor: 'background.default',
          minHeight: '100vh',
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

export default function AppLayout(props: AppLayoutProps) {
  return (
    <UserProvider>
      <AppLayoutContent {...props} />
    </UserProvider>
  );
}
