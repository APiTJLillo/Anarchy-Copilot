import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Typography,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  IconButton,
  Chip,
  Paper,
  Divider,
  Grid,
  FormHelperText,
  Switch,
  FormControlLabel,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Add,
  Delete,
  Preview,
  Save,
  Cancel,
  Check,
  ContentCopy,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

// Available operators for filter conditions
const OPERATORS = [
  { value: 'equals', label: 'Equals' },
  { value: 'not_equals', label: 'Not Equals' },
  { value: 'contains', label: 'Contains' },
  { value: 'not_contains', label: 'Not Contains' },
  { value: 'starts_with', label: 'Starts With' },
  { value: 'ends_with', label: 'Ends With' },
  { value: 'regex', label: 'Regex Match' },
  { value: 'greater_than', label: 'Greater Than' },
  { value: 'less_than', label: 'Less Than' },
  { value: 'in_list', label: 'In List' },
  { value: 'not_in_list', label: 'Not In List' },
];

// Common fields for filter conditions
const COMMON_FIELDS = [
  // Request fields
  { value: 'method', label: 'Request Method', type: 'request' },
  { value: 'path', label: 'Request Path', type: 'request' },
  { value: 'headers.User-Agent', label: 'User-Agent Header', type: 'request' },
  { value: 'headers.Content-Type', label: 'Content-Type Header', type: 'request' },
  { value: 'headers.Authorization', label: 'Authorization Header', type: 'request' },
  { value: 'headers.Cookie', label: 'Cookie Header', type: 'request' },
  { value: 'query_params', label: 'Query Parameters', type: 'request' },
  { value: 'body', label: 'Request Body', type: 'request' },
  
  // Response fields
  { value: 'status_code', label: 'Response Status Code', type: 'response' },
  { value: 'request_method', label: 'Original Request Method', type: 'response' },
  { value: 'request_path', label: 'Original Request Path', type: 'response' },
  { value: 'headers.Content-Type', label: 'Response Content-Type', type: 'response' },
  { value: 'headers.Server', label: 'Server Header', type: 'response' },
  { value: 'body', label: 'Response Body', type: 'response' },
];

/**
 * Component for editing filter rules with preview capability
 */
const FilterRuleEditor = ({ 
  rule = null, 
  onSave, 
  onCancel, 
  onPreview,
  isSaving = false,
  isPreviewing = false,
  previewResult = null,
}) => {
  const theme = useTheme();
  
  // Initialize state with rule data or defaults
  const [name, setName] = useState(rule?.name || '');
  const [description, setDescription] = useState(rule?.description || '');
  const [conditions, setConditions] = useState(rule?.conditions || []);
  const [enabled, setEnabled] = useState(rule?.enabled !== false);
  const [priority, setPriority] = useState(rule?.priority || 0);
  const [tags, setTags] = useState(rule?.tags || []);
  const [newTag, setNewTag] = useState('');
  
  // Form validation
  const [errors, setErrors] = useState({});
  
  // Initialize with rule data when it changes
  useEffect(() => {
    if (rule) {
      setName(rule.name || '');
      setDescription(rule.description || '');
      setConditions(rule.conditions || []);
      setEnabled(rule.enabled !== false);
      setPriority(rule.priority || 0);
      setTags(rule.tags || []);
    }
  }, [rule]);
  
  const validateForm = () => {
    const newErrors = {};
    
    if (!name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (conditions.length === 0) {
      newErrors.conditions = 'At least one condition is required';
    } else {
      // Validate each condition
      const conditionErrors = [];
      conditions.forEach((condition, index) => {
        const condError = {};
        if (!condition.field) {
          condError.field = 'Field is required';
        }
        if (!condition.operator) {
          condError.operator = 'Operator is required';
        }
        if (condition.value === undefined || condition.value === null || condition.value === '') {
          condError.value = 'Value is required';
        }
        
        if (Object.keys(condError).length > 0) {
          conditionErrors[index] = condError;
        }
      });
      
      if (conditionErrors.length > 0) {
        newErrors.conditionErrors = conditionErrors;
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleAddCondition = () => {
    setConditions([
      ...conditions,
      { field: '', operator: 'equals', value: '' }
    ]);
  };
  
  const handleRemoveCondition = (index) => {
    const newConditions = [...conditions];
    newConditions.splice(index, 1);
    setConditions(newConditions);
  };
  
  const handleConditionChange = (index, field, value) => {
    const newConditions = [...conditions];
    newConditions[index] = {
      ...newConditions[index],
      [field]: value
    };
    setConditions(newConditions);
    
    // Clear field-specific error if it exists
    if (errors.conditionErrors && errors.conditionErrors[index] && errors.conditionErrors[index][field]) {
      const newErrors = { ...errors };
      delete newErrors.conditionErrors[index][field];
      if (Object.keys(newErrors.conditionErrors[index]).length === 0) {
        newErrors.conditionErrors.splice(index, 1);
        if (newErrors.conditionErrors.length === 0) {
          delete newErrors.conditionErrors;
        }
      }
      setErrors(newErrors);
    }
  };
  
  const handleAddTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      setTags([...tags, newTag.trim()]);
      setNewTag('');
    }
  };
  
  const handleRemoveTag = (tagToRemove) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };
  
  const handleSave = () => {
    if (validateForm()) {
      const ruleData = {
        id: rule?.id,
        name,
        description,
        conditions,
        enabled,
        priority,
        tags,
      };
      
      onSave(ruleData);
    }
  };
  
  const handlePreview = () => {
    if (validateForm() && onPreview) {
      const ruleData = {
        id: rule?.id,
        name,
        description,
        conditions,
        enabled,
        priority,
        tags,
      };
      
      onPreview(ruleData);
    }
  };
  
  return (
    <Card variant="outlined">
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6" component="div">
            {rule?.id ? 'Edit Filter Rule' : 'Create Filter Rule'}
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
                color="primary"
              />
            }
            label={enabled ? 'Enabled' : 'Disabled'}
          />
        </Box>
        
        <Grid container spacing={2}>
          {/* Basic Information */}
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Rule Name"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                if (errors.name) {
                  setErrors({ ...errors, name: undefined });
                }
              }}
              margin="normal"
              error={!!errors.name}
              helperText={errors.name}
              required
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Priority"
              type="number"
              value={priority}
              onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
              margin="normal"
              helperText="Higher values have higher priority"
            />
          </Grid>
          
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              margin="normal"
              multiline
              rows={2}
            />
          </Grid>
          
          {/* Tags */}
          <Grid item xs={12}>
            <Box mb={1}>
              <Typography variant="subtitle2" gutterBottom>
                Tags
              </Typography>
              
              <Box display="flex" flexWrap="wrap" gap={1} mb={1}>
                {tags.map((tag, index) => (
                  <Chip
                    key={index}
                    label={tag}
                    onDelete={() => handleRemoveTag(tag)}
                    size="small"
                  />
                ))}
                
                {tags.length === 0 && (
                  <Typography variant="body2" color="text.secondary">
                    No tags added
                  </Typography>
                )}
              </Box>
              
              <Box display="flex" alignItems="center">
                <TextField
                  size="small"
                  label="Add Tag"
                  value={newTag}
                  onChange={(e) => setNewTag(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddTag();
                    }
                  }}
                  sx={{ mr: 1 }}
                />
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleAddTag}
                  startIcon={<Add />}
                >
                  Add
                </Button>
              </Box>
            </Box>
          </Grid>
          
          {/* Conditions */}
          <Grid item xs={12}>
            <Box mb={2}>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                <Typography variant="subtitle2">
                  Conditions
                </Typography>
                
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleAddCondition}
                  startIcon={<Add />}
                >
                  Add Condition
                </Button>
              </Box>
              
              {errors.conditions && (
                <FormHelperText error>{errors.conditions}</FormHelperText>
              )}
              
              {conditions.length === 0 ? (
                <Paper variant="outlined" sx={{ p: 2, bgcolor: theme.palette.background.default }}>
                  <Typography variant="body2" color="text.secondary" align="center">
                    No conditions added. Click "Add Condition" to create one.
                  </Typography>
                </Paper>
              ) : (
                conditions.map((condition, index) => {
                  const conditionErrors = errors.conditionErrors && errors.conditionErrors[index];
                  
                  return (
                    <Paper 
                      key={index} 
                      variant="outlined" 
                      sx={{ p: 2, mb: 2, bgcolor: theme.palette.background.default }}
                    >
                      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                        <Typography variant="body2" fontWeight="medium">
                          Condition {index + 1}
                        </Typography>
                        
                        <IconButton 
                          size="small" 
                          onClick={() => handleRemoveCondition(index)}
                          color="error"
                        >
                          <Delete fontSize="small" />
                        </IconButton>
                      </Box>
                      
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={4}>
                          <FormControl 
                            fullWidth 
                            size="small" 
                            error={!!(conditionErrors && conditionErrors.field)}
                          >
                            <InputLabel>Field</InputLabel>
                            <Select
                              value={condition.field}
                              onChange={(e) => handleConditionChange(index, 'field', e.target.value)}
                              label="Field"
                            >
                              {COMMON_FIELDS.map((field) => (
                                <MenuItem key={field.value} value={field.value}>
                                  {field.label}
                                </MenuItem>
                              ))}
                              <MenuItem value="custom">
                                <em>Custom Field...</em>
                              </MenuItem>
                            </Select>
                            {conditionErrors && conditionErrors.field && (
                              <FormHelperText>{conditionErrors.field}</FormHelperText>
                            )}
                          </FormControl>
                          
                          {condition.field === 'custom' && (
                            <TextField
                              fullWidth
                              size="small"
                              label="Custom Field"
                              value={condition.customField || ''}
                              onChange={(e) => handleConditionChange(index, 'customField', e.target.value)}
                              margin="normal"
                              placeholder="e.g., headers.X-Custom-Header"
                            />
                          )}
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <FormControl 
                            fullWidth 
                            size="small"
                            error={!!(conditionErrors && conditionErrors.operator)}
                          >
                            <InputLabel>Operator</InputLabel>
                            <Select
                              value={condition.operator}
                              onChange={(e) => handleConditionChange(index, 'operator', e.target.value)}
                              label="Operator"
                            >
                              {OPERATORS.map((op) => (
                                <MenuItem key={op.value} value={op.value}>
                                  {op.label}
                                </MenuItem>
                              ))}
                            </Select>
                            {conditionErrors && conditionErrors.operator && (
                              <FormHelperText>{conditionErrors.operator}</FormHelperText>
                            )}
                          </FormControl>
                        </Grid>
                        
                        <Grid item xs={12} md={4}>
                          <TextField
                            fullWidth
                            size="small"
                            label="Value"
                            value={condition.value}
                            onChange={(e) => handleConditionChange(index, 'value', e.target.value)}
                            error={!!(conditionErrors && conditionErrors.value)}
                            helperText={conditionErrors && conditionErrors.value}
                          />
                        </Grid>
                      </Grid>
                    </Paper>
                  );
                })
              )}
            </Box>
          </Grid>
        </Grid>
        
        {/* Preview Result */}
        {previewResult && (
          <Box mt={3}>
            <Typography variant="subtitle2" gutterBottom>
              Preview Result
            </Typography>
            
            <Paper variant="outlined" sx={{ p: 2, bgcolor: theme.palette.background.default }}>
              <Typography variant="body2" fontWeight="medium" gutterBottom>
                Would {previewResult.would_match ? 'match' : 'not match'} {previewResult.traffic_count} traffic items
              </Typography>
              
              {previewResult.sample_matches && previewResult.sample_matches.length > 0 && (
                <Box mt={1}>
                  <Typography variant="body2" gutterBottom>
                    Sample matches:
                  </Typography>
                  
                  <Box component="ul" sx={{ mt: 0, pl: 2 }}>
                    {previewResult.sample_matches.map((match, idx) => (
                      <Box component="li" key={idx}>
                        <Typography variant="body2" color="text.secondary">
                          {match.type === 'request' 
                            ? `${match.method} ${match.path}`
                            : `${match.status_code} response for ${match.request_path}`
                          }
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}
            </Paper>
          </Box>
        )}
        
        {/* Action Buttons */}
        <Box display="flex" justifyContent="flex-end" mt={3} gap={1}>
          <Button
            variant="outlined"
            onClick={onCancel}
            startIcon={<Cancel />}
          >
            Cancel
          </Button>
          
          <Button
            variant="outlined"
            onClick={handlePreview}
            startIcon={<Preview />}
            disabled={isPreviewing}
          >
            {isPreviewing ? (
              <>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                Previewing...
              </>
            ) : 'Preview'}
          </Button>
          
          <Button
            variant="contained"
            onClick={handleSave}
            startIcon={<Save />}
            disabled={isSaving}
          >
            {isSaving ? (
              <>
                <CircularProgress size={20} sx={{ mr: 1 }} />
                Saving...
              </>
            ) : 'Save Rule'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default FilterRuleEditor;
