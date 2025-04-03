import React, { useState, useEffect } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    TextField,
    Typography,
    Box,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    List,
    ListItem,
    ListItemText,
    IconButton,
    Divider,
    Paper,
    Grid,
    CircularProgress,
    Chip
} from '@mui/material';
import {
    Add as AddIcon,
    Delete as DeleteIcon,
    Upload as UploadIcon,
    Download as DownloadIcon
} from '@mui/icons-material';
import axios from 'axios';

interface FuzzingList {
    id: string;
    name: string;
    description: string;
    category: string;
    payload_count: number;
    created_at: string;
    updated_at: string;
}

interface FuzzingConfigPanelProps {
    connectionId: string;
    onClose: () => void;
    open: boolean;
}

export const FuzzingConfigPanel: React.FC<FuzzingConfigPanelProps> = ({ connectionId, onClose, open }) => {
    const [lists, setLists] = useState<FuzzingList[]>([]);
    const [selectedList, setSelectedList] = useState<string>('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    
    // New list form state
    const [showNewListForm, setShowNewListForm] = useState(false);
    const [newListName, setNewListName] = useState('');
    const [newListDescription, setNewListDescription] = useState('');
    const [newListCategory, setNewListCategory] = useState('custom');
    const [newListPayloads, setNewListPayloads] = useState<string[]>([]);
    const [newPayload, setNewPayload] = useState('');
    
    // File upload state
    const [showFileUpload, setShowFileUpload] = useState(false);
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [uploadName, setUploadName] = useState('');
    const [uploadCategory, setUploadCategory] = useState('imported');
    
    // Load fuzzing lists
    useEffect(() => {
        if (open) {
            loadFuzzingLists();
        }
    }, [open]);
    
    const loadFuzzingLists = async () => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await axios.get('/api/proxy/websocket/fuzzing/lists');
            setLists(response.data);
            if (response.data.length > 0) {
                setSelectedList(response.data[0].id);
            }
        } catch (err) {
            console.error('Failed to load fuzzing lists:', err);
            setError('Failed to load fuzzing lists. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    const handleStartFuzzing = async () => {
        if (!selectedList) {
            setError('Please select a fuzzing list');
            return;
        }
        
        setLoading(true);
        setError(null);
        setSuccess(null);
        
        try {
            await axios.post(`/api/proxy/websocket/fuzzing/start/${connectionId}`, {
                list_id: selectedList
            });
            setSuccess('Fuzzing started successfully');
            setTimeout(() => {
                onClose();
            }, 1500);
        } catch (err) {
            console.error('Failed to start fuzzing:', err);
            setError('Failed to start fuzzing. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    const handleCreateList = async () => {
        if (!newListName) {
            setError('Please enter a name for the list');
            return;
        }
        
        if (newListPayloads.length === 0) {
            setError('Please add at least one payload to the list');
            return;
        }
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await axios.post('/api/proxy/websocket/fuzzing/lists', {
                name: newListName,
                description: newListDescription,
                category: newListCategory,
                payloads: newListPayloads
            });
            
            setLists([...lists, response.data]);
            setSelectedList(response.data.id);
            setShowNewListForm(false);
            setNewListName('');
            setNewListDescription('');
            setNewListCategory('custom');
            setNewListPayloads([]);
            setSuccess('Fuzzing list created successfully');
        } catch (err) {
            console.error('Failed to create fuzzing list:', err);
            setError('Failed to create fuzzing list. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    const handleAddPayload = () => {
        if (!newPayload) return;
        setNewListPayloads([...newListPayloads, newPayload]);
        setNewPayload('');
    };
    
    const handleRemovePayload = (index: number) => {
        const updatedPayloads = [...newListPayloads];
        updatedPayloads.splice(index, 1);
        setNewListPayloads(updatedPayloads);
    };
    
    const handleFileUpload = async () => {
        if (!uploadFile) {
            setError('Please select a file to upload');
            return;
        }
        
        if (!uploadName) {
            setUploadName(uploadFile.name.split('.')[0]);
        }
        
        setLoading(true);
        setError(null);
        
        try {
            const formData = new FormData();
            formData.append('file', uploadFile);
            formData.append('name', uploadName);
            formData.append('category', uploadCategory);
            
            const response = await axios.post('/api/proxy/websocket/fuzzing/lists/import', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            
            setLists([...lists, response.data]);
            setSelectedList(response.data.id);
            setShowFileUpload(false);
            setUploadFile(null);
            setUploadName('');
            setUploadCategory('imported');
            setSuccess('Fuzzing list imported successfully');
        } catch (err) {
            console.error('Failed to import fuzzing list:', err);
            setError('Failed to import fuzzing list. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    const handleDeleteList = async (listId: string) => {
        if (!window.confirm('Are you sure you want to delete this fuzzing list?')) {
            return;
        }
        
        setLoading(true);
        setError(null);
        
        try {
            await axios.delete(`/api/proxy/websocket/fuzzing/lists/${listId}`);
            setLists(lists.filter(list => list.id !== listId));
            if (selectedList === listId) {
                setSelectedList(lists.length > 1 ? lists.filter(list => list.id !== listId)[0].id : '');
            }
            setSuccess('Fuzzing list deleted successfully');
        } catch (err) {
            console.error('Failed to delete fuzzing list:', err);
            setError('Failed to delete fuzzing list. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>WebSocket Fuzzing Configuration</DialogTitle>
            <DialogContent>
                {loading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                        <CircularProgress />
                    </Box>
                )}
                
                {error && (
                    <Box sx={{ my: 2 }}>
                        <Typography color="error">{error}</Typography>
                    </Box>
                )}
                
                {success && (
                    <Box sx={{ my: 2 }}>
                        <Typography color="success.main">{success}</Typography>
                    </Box>
                )}
                
                {!showNewListForm && !showFileUpload && (
                    <Box sx={{ my: 2 }}>
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                    <Typography variant="h6">Available Fuzzing Lists</Typography>
                                    <Box>
                                        <Button 
                                            startIcon={<AddIcon />} 
                                            variant="outlined" 
                                            onClick={() => setShowNewListForm(true)}
                                            sx={{ mr: 1 }}
                                        >
                                            Create New
                                        </Button>
                                        <Button 
                                            startIcon={<UploadIcon />} 
                                            variant="outlined" 
                                            onClick={() => setShowFileUpload(true)}
                                        >
                                            Import
                                        </Button>
                                    </Box>
                                </Box>
                            </Grid>
                            
                            <Grid item xs={12}>
                                {lists.length === 0 ? (
                                    <Paper sx={{ p: 2 }}>
                                        <Typography align="center">No fuzzing lists available. Create a new one or import from a file.</Typography>
                                    </Paper>
                                ) : (
                                    <FormControl fullWidth>
                                        <InputLabel id="fuzzing-list-select-label">Select Fuzzing List</InputLabel>
                                        <Select
                                            labelId="fuzzing-list-select-label"
                                            value={selectedList}
                                            label="Select Fuzzing List"
                                            onChange={(e) => setSelectedList(e.target.value)}
                                        >
                                            {lists.map((list) => (
                                                <MenuItem key={list.id} value={list.id}>
                                                    {list.name} ({list.payload_count} payloads) - {list.category}
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                )}
                            </Grid>
                            
                            {selectedList && (
                                <Grid item xs={12}>
                                    <Paper sx={{ p: 2, mt: 2 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <Typography variant="subtitle1">
                                                {lists.find(l => l.id === selectedList)?.name}
                                            </Typography>
                                            <IconButton 
                                                color="error" 
                                                onClick={() => handleDeleteList(selectedList)}
                                                title="Delete list"
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        </Box>
                                        <Typography variant="body2" color="textSecondary">
                                            {lists.find(l => l.id === selectedList)?.description}
                                        </Typography>
                                        <Box sx={{ mt: 1 }}>
                                            <Chip 
                                                label={lists.find(l => l.id === selectedList)?.category} 
                                                size="small" 
                                                sx={{ mr: 1 }}
                                            />
                                            <Chip 
                                                label={`${lists.find(l => l.id === selectedList)?.payload_count} payloads`} 
                                                size="small" 
                                            />
                                        </Box>
                                    </Paper>
                                </Grid>
                            )}
                        </Grid>
                    </Box>
                )}
                
                {showNewListForm && (
                    <Box sx={{ my: 2 }}>
                        <Typography variant="h6" gutterBottom>Create New Fuzzing List</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    label="List Name"
                                    fullWidth
                                    value={newListName}
                                    onChange={(e) => setNewListName(e.target.value)}
                                    required
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <FormControl fullWidth>
                                    <InputLabel>Category</InputLabel>
                                    <Select
                                        value={newListCategory}
                                        label="Category"
                                        onChange={(e) => setNewListCategory(e.target.value)}
                                    >
                                        <MenuItem value="custom">Custom</MenuItem>
                                        <MenuItem value="sql_injection">SQL Injection</MenuItem>
                                        <MenuItem value="xss">XSS</MenuItem>
                                        <MenuItem value="protocol">Protocol</MenuItem>
                                        <MenuItem value="json">JSON</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    label="Description"
                                    fullWidth
                                    multiline
                                    rows={2}
                                    value={newListDescription}
                                    onChange={(e) => setNewListDescription(e.target.value)}
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <Typography variant="subtitle1" gutterBottom>Payloads</Typography>
                                <Box sx={{ display: 'flex', mb: 2 }}>
                                    <TextField
                                        label="Add Payload"
                                        fullWidth
                                        value={newPayload}
                                        onChange={(e) => setNewPayload(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && handleAddPayload()}
                                    />
                                    <Button
                                        variant="contained"
                                        onClick={handleAddPayload}
                                        sx={{ ml: 1 }}
                                    >
                                        Add
                                    </Button>
                                </Box>
                                <Paper sx={{ maxHeight: 200, overflow: 'auto', p: 1 }}>
                                    <List dense>
                                        {newListPayloads.map((payload, index) => (
                                            <ListItem
                                                key={index}
                                                secondaryAction={
                                                    <IconButton edge="end" onClick={() => handleRemovePayload(index)}>
                                                        <DeleteIcon />
                                                    </IconButton>
                                                }
                                            >
                                                <ListItemText primary={payload} />
                                            </ListItem>
                                        ))}
                                        {newListPayloads.length === 0 && (
                                            <ListItem>
                                                <ListItemText primary="No payloads added yet" />
                                            </ListItem>
                                        )}
                                    </List>
                                </Paper>
                            </Grid>
                        </Grid>
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                            <Button onClick={() => setShowNewListForm(false)} sx={{ mr: 1 }}>
                                Cancel
                            </Button>
                            <Button 
                                variant="contained" 
                                onClick={handleCreateList}
                                disabled={!newListName || newListPayloads.length === 0}
                            >
                                Create List
                            </Button>
                        </Box>
                    </Box>
                )}
                
                {showFileUpload && (
                    <Box sx={{ my: 2 }}>
                        <Typography variant="h6" gutterBottom>Import Fuzzing List from File</Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
                                <Button
                                    variant="outlined"
                                    component="label"
                                    fullWidth
                                    sx={{ height: 56, borderStyle: 'dashed' }}
                                >
                                    {uploadFile ? uploadFile.name : 'Select File'}
                                    <input
                                        type="file"
                                        hidden
                                        onChange={(e) => {
                                            if (e.target.files && e.target.files[0]) {
                                                setUploadFile(e.target.files[0]);
                                                setUploadName(e.target.files[0].name.split('.')[0]);
                                            }
                                        }}
                                    />
                                </Button>
                                <Typography variant="caption" color="textSecondary">
                                    Upload a text file with one payload per line
                                </Typography>
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    label="List Name"
                                    fullWidth
                                    value={uploadName}
                                    onChange={(e) => setUploadName(e.target.value)}
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <FormControl fullWidth>
                                    <InputLabel>Category</InputLabel>
                                    <Select
                                        value={uploadCategory}
                                        label="Category"
                                        onChange={(e) => setUploadCategory(e.target.value)}
                                    >
                                        <MenuItem value="imported">Imported</MenuItem>
                                        <MenuItem value="sql_injection">SQL Injection</MenuItem>
                                        <MenuItem value="xss">XSS</MenuItem>
                                        <MenuItem value="protocol">Protocol</MenuItem>
                                        <MenuItem value="json">JSON</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                        </Grid>
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                            <Button onClick={() => setShowFileUpload(false)} sx={{ mr: 1 }}>
                                Cancel
                            </Button>
                            <Button 
                                variant="contained" 
                                onClick={handleFileUpload}
                                disabled={!uploadFile}
                            >
                                Import
                            </Button>
                        </Box>
                    </Box>
                )}
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button 
                    variant="contained" 
                    onClick={handleStartFuzzing}
                    disabled={!selectedList || lists.length === 0 || showNewListForm || showFileUpload}
                >
                    Start Fuzzing
                </Button>
            </DialogActions>
        </Dialog>
    );
};
