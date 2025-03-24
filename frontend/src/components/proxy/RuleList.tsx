import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    IconButton,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemSecondaryAction,
    ListItemText,
    Stack,
    Switch,
    Tooltip,
    Typography,
} from '@mui/material';
import ConfirmDialog from '../shared/ConfirmDialog';
import {
    DragHandle as DragHandleIcon,
    Edit as EditIcon,
    Delete as DeleteIcon,
    Block as BlockIcon,
    SwapHoriz as ForwardIcon,
    Edit as ModifyIcon,
} from '@mui/icons-material';
import {
    DragDropContext,
    Droppable,
    Draggable,
    DropResult,
    DroppableProvided,
    DraggableProvided
} from 'react-beautiful-dnd';
import { InterceptionRule, ActionType } from '../../api/proxyApi';

interface RuleListProps {
    rules: InterceptionRule[];
    onEditRule: (rule: InterceptionRule) => void;
    onDeleteRule: (ruleId: number) => void;
    onToggleRule: (ruleId: number, enabled: boolean) => void;
    onReorderRules: (ruleIds: number[]) => void;
}

const ACTION_ICONS: Record<ActionType, typeof BlockIcon> = {
    'BLOCK': BlockIcon,
    'FORWARD': ForwardIcon,
    'MODIFY': ModifyIcon,
};

const RuleList: React.FC<RuleListProps> = ({
    rules,
    onEditRule,
    onDeleteRule,
    onToggleRule,
    onReorderRules,
}) => {
    const [orderedRules, setOrderedRules] = useState(rules);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [ruleToDelete, setRuleToDelete] = useState<number | null>(null);

    useEffect(() => {
        setOrderedRules(rules);
    }, [rules]);

    const handleDragEnd = (result: DropResult) => {
        if (!result.destination) return;

        const items = Array.from(orderedRules);
        const [reorderedItem] = items.splice(result.source.index, 1);
        items.splice(result.destination.index, 0, reorderedItem);

        setOrderedRules(items);
        onReorderRules(items.map(rule => rule.id));
    };

    const handleDeleteConfirm = useCallback(() => {
        if (ruleToDelete !== null) {
            onDeleteRule(ruleToDelete);
            setDeleteDialogOpen(false);
            setRuleToDelete(null);
        }
    }, [ruleToDelete, onDeleteRule]);

    const handleDeleteCancel = useCallback(() => {
        setDeleteDialogOpen(false);
        setRuleToDelete(null);
    }, []);

    const getConditionSummary = (rule: InterceptionRule) => {
        return rule.conditions.map(c => {
            const operator = c.operator.replace('_', ' ');
            return `${c.field} ${operator} "${c.value}"${c.use_regex ? ' (regex)' : ''}`;
        }).join(' AND ');
    };

    return (
        <Card>
            <CardContent>
                <Stack spacing={2}>
                    <Typography variant="h6" gutterBottom>
                        Interception Rules
                    </Typography>

                    <DragDropContext onDragEnd={handleDragEnd}>
                        <Droppable droppableId="rules">
                            {(provided: DroppableProvided) => (
                                <List
                                    {...provided.droppableProps}
                                    ref={provided.innerRef}
                                    sx={{
                                        width: '100%',
                                        bgcolor: 'background.paper',
                                        '& .MuiListItem-root': {
                                            mb: 1,
                                            bgcolor: 'background.default',
                                            borderRadius: 1,
                                        },
                                    }}
                                >
                                    {orderedRules.map((rule, index) => {
                                        const ActionIcon = ACTION_ICONS[rule.action.type];

                                        return (
                                            <Draggable
                                                key={rule.id}
                                                draggableId={rule.id.toString()}
                                                index={index}
                                            >
                                                {(provided: DraggableProvided) => (
                                                    <ListItem
                                                        ref={provided.innerRef}
                                                        {...provided.draggableProps}
                                                    >
                                                        <ListItemIcon {...provided.dragHandleProps}>
                                                            <DragHandleIcon />
                                                        </ListItemIcon>

                                                        <ListItemIcon>
                                                            <ActionIcon
                                                                sx={{
                                                                    color: rule.enabled ? 'primary.main' : 'action.disabled'
                                                                }}
                                                            />
                                                        </ListItemIcon>

                                                        <ListItemText
                                                            primary={rule.name}
                                                            secondary={
                                                                <Box component="span" sx={{ fontSize: '0.8rem' }}>
                                                                    {getConditionSummary(rule)}
                                                                </Box>
                                                            }
                                                            sx={{
                                                                opacity: rule.enabled ? 1 : 0.5,
                                                            }}
                                                        />

                                                        <ListItemSecondaryAction>
                                                            <Stack direction="row" spacing={1}>
                                                                <Switch
                                                                    checked={rule.enabled}
                                                                    onChange={(e) =>
                                                                        onToggleRule(rule.id, e.target.checked)
                                                                    }
                                                                />
                                                                <Tooltip title="Edit Rule">
                                                                    <IconButton
                                                                        onClick={() => onEditRule(rule)}
                                                                        size="small"
                                                                    >
                                                                        <EditIcon />
                                                                    </IconButton>
                                                                </Tooltip>
                                                                <Tooltip title="Delete Rule">
                                                                    <IconButton
                                                                        onClick={(e) => {
                                                                            e.stopPropagation();
                                                                            setRuleToDelete(rule.id);
                                                                            setDeleteDialogOpen(true);
                                                                        }}
                                                                        size="small"
                                                                        color="error"
                                                                    >
                                                                        <DeleteIcon />
                                                                    </IconButton>
                                                                </Tooltip>
                                                            </Stack>
                                                        </ListItemSecondaryAction>
                                                    </ListItem>
                                                )}
                                            </Draggable>
                                        );
                                    })}
                                    {provided.placeholder}
                                </List>
                            )}
                        </Droppable>
                    </DragDropContext>

                    {orderedRules.length === 0 && (
                        <Typography variant="body2" color="text.secondary" align="center">
                            No rules defined yet
                        </Typography>
                    )}
                </Stack>
            </CardContent>

            <ConfirmDialog
                open={deleteDialogOpen}
                title="Delete Rule"
                message="Are you sure you want to delete this rule? This action cannot be undone."
                onConfirm={handleDeleteConfirm}
                onCancel={handleDeleteCancel}
                confirmText="Delete"
                cancelText="Cancel"
                severity="error"
            />
        </Card>
    );
};

export default RuleList;
