import React, { useState, useEffect } from 'react';
import { Box, Button, Dialog, Stack } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RuleEditor from './RuleEditor';
import RuleList from './RuleList';
import proxyApi, { InterceptionRule, CreateRuleRequest } from '../../api/proxyApi';

interface InterceptionRuleManagerProps {
    sessionId: number;
}

const InterceptionRuleManager: React.FC<InterceptionRuleManagerProps> = ({ sessionId }) => {
    const [rules, setRules] = useState<InterceptionRule[]>([]);
    const [isEditorOpen, setIsEditorOpen] = useState(false);
    const [editingRule, setEditingRule] = useState<InterceptionRule | null>(null);

    const loadRules = async () => {
        try {
            const data = await proxyApi.getRules(sessionId);
            setRules(data);
        } catch (error) {
            console.error('Failed to load rules:', error);
        }
    };

    useEffect(() => {
        loadRules();
    }, [sessionId]);

    const handleCreateRule = async (rule: CreateRuleRequest) => {
        try {
            await proxyApi.createRule(sessionId, rule);
            setIsEditorOpen(false);
            loadRules();
        } catch (error) {
            console.error('Failed to create rule:', error);
        }
    };

    const handleUpdateRule = async (rule: CreateRuleRequest) => {
        try {
            if (!editingRule) return;
            await proxyApi.updateRule(sessionId, editingRule.id, rule);
            setIsEditorOpen(false);
            setEditingRule(null);
            loadRules();
        } catch (error) {
            console.error('Failed to update rule:', error);
        }
    };

    const handleDeleteRule = async (ruleId: number) => {
        try {
            await proxyApi.deleteRule(sessionId, ruleId);
            loadRules();
        } catch (error) {
            console.error('Failed to delete rule:', error);
        }
    };

    const handleToggleRule = async (ruleId: number, enabled: boolean) => {
        try {
            await proxyApi.updateRule(sessionId, ruleId, { enabled });
            loadRules();
        } catch (error) {
            console.error('Failed to toggle rule:', error);
        }
    };

    const handleReorderRules = async (ruleIds: number[]) => {
        try {
            await proxyApi.reorderRules(sessionId, ruleIds);
            loadRules();
        } catch (error) {
            console.error('Failed to reorder rules:', error);
        }
    };

    const handleOpenEditor = (rule?: InterceptionRule) => {
        setEditingRule(rule || null);
        setIsEditorOpen(true);
    };

    const handleCloseEditor = () => {
        setIsEditorOpen(false);
        setEditingRule(null);
    };

    return (
        <Stack spacing={2}>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => handleOpenEditor()}
                >
                    Add Rule
                </Button>
            </Box>

            <RuleList
                rules={rules}
                onEditRule={handleOpenEditor}
                onDeleteRule={handleDeleteRule}
                onToggleRule={handleToggleRule}
                onReorderRules={handleReorderRules}
            />

            <Dialog
                open={isEditorOpen}
                onClose={handleCloseEditor}
                maxWidth="md"
                fullWidth
            >
                <RuleEditor
                    onSave={editingRule ? handleUpdateRule : handleCreateRule}
                    onCancel={handleCloseEditor}
                    initialValue={editingRule || undefined}
                />
            </Dialog>
        </Stack>
    );
};

export default InterceptionRuleManager;
