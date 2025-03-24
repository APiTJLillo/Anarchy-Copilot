import React, { useState } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    FormControl,
    IconButton,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Switch,
    TextField,
    Typography,
    FormControlLabel,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import {
    Condition,
    CreateRuleRequest,
    Modification,
    ActionType,
    RuleAction
} from '../../api/proxyApi';

interface RuleEditorProps {
    onSave: (rule: CreateRuleRequest) => void;
    onCancel: () => void;
    initialValue?: Partial<CreateRuleRequest>;
}

const FIELD_OPTIONS = [
    { value: 'url', label: 'URL' },
    { value: 'method', label: 'Method' },
    { value: 'request_headers', label: 'Request Headers' },
    { value: 'request_body', label: 'Request Body' },
    { value: 'status_code', label: 'Status Code' },
    { value: 'response_headers', label: 'Response Headers' },
    { value: 'response_body', label: 'Response Body' },
];

const OPERATOR_OPTIONS = [
    { value: 'equals', label: 'Equals' },
    { value: 'not_equals', label: 'Not Equals' },
    { value: 'contains', label: 'Contains' },
    { value: 'not_contains', label: 'Not Contains' },
    { value: 'starts_with', label: 'Starts With' },
    { value: 'ends_with', label: 'Ends With' },
];

const ACTION_OPTIONS = [
    { value: 'FORWARD', label: 'Forward' },
    { value: 'BLOCK', label: 'Block' },
    { value: 'MODIFY', label: 'Modify' },
] as const;

export const RuleEditor: React.FC<RuleEditorProps> = ({
    onSave,
    onCancel,
    initialValue = {}
}) => {
    const [name, setName] = useState(initialValue.name || '');
    const [enabled, setEnabled] = useState(initialValue.enabled ?? true);
    const [conditions, setConditions] = useState<Condition[]>(
        initialValue.conditions || [{ field: 'url', operator: 'contains', value: '', use_regex: false }]
    );
    const [actionType, setActionType] = useState<ActionType>(
        (initialValue.action?.type as ActionType) || 'FORWARD'
    );
    const [modification, setModification] = useState<Modification>({
        field: '',
        value: '',
        headers: {},
        ...(initialValue.action?.modifications?.[0] || {})
    });

    const handleAddCondition = () => {
        setConditions([
            ...conditions,
            { field: 'url', operator: 'contains', value: '', use_regex: false }
        ]);
    };

    const handleRemoveCondition = (index: number) => {
        setConditions(conditions.filter((_, i) => i !== index));
    };

    const handleConditionChange = (index: number, field: keyof Condition, value: any) => {
        const newConditions = [...conditions];
        newConditions[index] = { ...newConditions[index], [field]: value };
        setConditions(newConditions);
    };

    const handleSave = () => {
        const rule: CreateRuleRequest = {
            name,
            enabled,
            conditions,
            modifications: actionType === 'MODIFY' ? [modification] : [],
            action: {
                type: actionType,
                modifications: actionType === 'MODIFY' ? [modification] : []
            },
            order: initialValue.order || 0,
            session_id: initialValue.session_id || 0
        };
        onSave(rule);
    };

    return (
        <Card>
            <CardContent>
                <Stack spacing={3}>
                    <Typography variant="h6">
                        {initialValue.name ? 'Edit Rule' : 'Create Rule'}
                    </Typography>

                    <Stack direction="row" spacing={2} alignItems="center">
                        <TextField
                            fullWidth
                            label="Rule Name"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                        />
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={enabled}
                                    onChange={(e) => setEnabled(e.target.checked)}
                                />
                            }
                            label="Enabled"
                        />
                    </Stack>

                    <Box>
                        <Typography variant="subtitle1" gutterBottom>
                            Conditions
                        </Typography>
                        {conditions.map((condition, index) => (
                            <Stack
                                key={index}
                                direction="row"
                                spacing={2}
                                mb={2}
                                alignItems="center"
                            >
                                <FormControl fullWidth>
                                    <InputLabel>Field</InputLabel>
                                    <Select
                                        value={condition.field}
                                        label="Field"
                                        onChange={(e) =>
                                            handleConditionChange(index, 'field', e.target.value)
                                        }
                                    >
                                        {FIELD_OPTIONS.map(option => (
                                            <MenuItem key={option.value} value={option.value}>
                                                {option.label}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>

                                <FormControl fullWidth>
                                    <InputLabel>Operator</InputLabel>
                                    <Select
                                        value={condition.operator}
                                        label="Operator"
                                        onChange={(e) =>
                                            handleConditionChange(index, 'operator', e.target.value)
                                        }
                                    >
                                        {OPERATOR_OPTIONS.map(option => (
                                            <MenuItem key={option.value} value={option.value}>
                                                {option.label}
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>

                                <TextField
                                    fullWidth
                                    label="Value"
                                    value={condition.value}
                                    onChange={(e) =>
                                        handleConditionChange(index, 'value', e.target.value)
                                    }
                                />

                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={condition.use_regex}
                                            onChange={(e) =>
                                                handleConditionChange(
                                                    index,
                                                    'use_regex',
                                                    e.target.checked
                                                )
                                            }
                                        />
                                    }
                                    label="Regex"
                                />

                                <IconButton
                                    onClick={() => handleRemoveCondition(index)}
                                    disabled={conditions.length === 1}
                                >
                                    <DeleteIcon />
                                </IconButton>
                            </Stack>
                        ))}
                        <Button
                            startIcon={<AddIcon />}
                            onClick={handleAddCondition}
                            variant="outlined"
                            size="small"
                        >
                            Add Condition
                        </Button>
                    </Box>

                    <FormControl fullWidth>
                        <InputLabel>Action</InputLabel>
                        <Select
                            value={actionType}
                            label="Action"
                            onChange={(e) => setActionType(e.target.value as ActionType)}
                        >
                            {ACTION_OPTIONS.map(option => (
                                <MenuItem key={option.value} value={option.value}>
                                    {option.label}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>

                    {actionType === 'MODIFY' && (
                        <Box>
                            <Typography variant="subtitle1" gutterBottom>
                                Modifications
                            </Typography>
                            <Stack spacing={2}>
                                <TextField
                                    fullWidth
                                    label="Headers (JSON)"
                                    multiline
                                    rows={3}
                                    value={
                                        modification.headers
                                            ? JSON.stringify(modification.headers, null, 2)
                                            : ''
                                    }
                                    onChange={(e) => {
                                        try {
                                            const headers = e.target.value
                                                ? JSON.parse(e.target.value)
                                                : {};
                                            setModification({ ...modification, headers });
                                        } catch { } // Ignore invalid JSON while typing
                                    }}
                                />
                                <TextField
                                    fullWidth
                                    label="Body"
                                    multiline
                                    rows={3}
                                    value={modification.body || ''}
                                    onChange={(e) =>
                                        setModification({
                                            ...modification,
                                            body: e.target.value
                                        })
                                    }
                                />
                                <TextField
                                    label="Status Code"
                                    type="number"
                                    value={modification.status_code || ''}
                                    onChange={(e) =>
                                        setModification({
                                            ...modification,
                                            status_code: parseInt(e.target.value) || undefined
                                        })
                                    }
                                />
                            </Stack>
                        </Box>
                    )}

                    <Stack direction="row" spacing={2} justifyContent="flex-end">
                        <Button onClick={onCancel} variant="outlined">
                            Cancel
                        </Button>
                        <Button
                            onClick={handleSave}
                            variant="contained"
                            disabled={!name || conditions.some(c => !c.value)}
                        >
                            Save
                        </Button>
                    </Stack>
                </Stack>
            </CardContent>
        </Card>
    );
};

export default RuleEditor;
