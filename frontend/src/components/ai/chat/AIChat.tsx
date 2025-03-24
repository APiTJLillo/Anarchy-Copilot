import React from 'react';
import {
    Box,
    Card,
    TextField,
    IconButton,
    Typography,
    List,
    ListItem,
    ListItemText,
    ListItemAvatar,
    Avatar,
    Divider,
    Paper,
} from '@mui/material';
import { Send as SendIcon, SmartToy as BotIcon } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../../store/hooks';

interface ChatMessage {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
}

const AIChat: React.FC = () => {
    const [message, setMessage] = React.useState('');
    const [chatHistory, setChatHistory] = React.useState<ChatMessage[]>([]);
    const inputRef = React.useRef<HTMLInputElement>(null);

    const handleSendMessage = (e: React.FormEvent) => {
        e.preventDefault();
        if (!message.trim()) return;

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            text: message,
            sender: 'user',
            timestamp: new Date()
        };

        setChatHistory(prev => [...prev, userMessage]);

        // Simulate bot response
        setTimeout(() => {
            const botMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                text: "I'm a placeholder response. Real AI integration coming soon!",
                sender: 'bot',
                timestamp: new Date()
            };
            setChatHistory(prev => [...prev, botMessage]);
        }, 1000);

        setMessage('');
    };

    React.useEffect(() => {
        if (chatHistory.length > 0) {
            inputRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [chatHistory]);

    return (
        <Box sx={{ height: '70vh', display: 'flex', flexDirection: 'column' }}>
            <Paper
                elevation={0}
                sx={{
                    flex: 1,
                    mb: 2,
                    p: 2,
                    overflowY: 'auto',
                    bgcolor: 'background.default'
                }}
            >
                {chatHistory.length === 0 ? (
                    <Box
                        display="flex"
                        alignItems="center"
                        justifyContent="center"
                        height="100%"
                    >
                        <Typography color="text.secondary">
                            Start a conversation with the AI assistant
                        </Typography>
                    </Box>
                ) : (
                    <List>
                        {chatHistory.map((msg, index) => (
                            <React.Fragment key={msg.id}>
                                <ListItem
                                    alignItems="flex-start"
                                    sx={{
                                        flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row'
                                    }}
                                >
                                    <ListItemAvatar>
                                        <Avatar sx={{
                                            bgcolor: msg.sender === 'user' ? 'primary.main' : 'secondary.main'
                                        }}>
                                            {msg.sender === 'user' ? 'U' : <BotIcon />}
                                        </Avatar>
                                    </ListItemAvatar>
                                    <ListItemText
                                        primary={msg.text}
                                        secondary={msg.timestamp.toLocaleTimeString()}
                                        sx={{
                                            textAlign: msg.sender === 'user' ? 'right' : 'left'
                                        }}
                                    />
                                </ListItem>
                                {index < chatHistory.length - 1 && (
                                    <Divider variant="inset" component="li" />
                                )}
                            </React.Fragment>
                        ))}
                    </List>
                )}
            </Paper>

            <Box
                component="form"
                onSubmit={handleSendMessage}
                sx={{
                    display: 'flex',
                    gap: 1
                }}
            >
                <TextField
                    fullWidth
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    variant="outlined"
                    inputRef={inputRef}
                    InputProps={{
                        sx: { bgcolor: 'background.paper' }
                    }}
                />
                <IconButton
                    type="submit"
                    color="primary"
                    sx={{
                        bgcolor: 'primary.main',
                        color: 'white',
                        '&:hover': {
                            bgcolor: 'primary.dark'
                        }
                    }}
                >
                    <SendIcon />
                </IconButton>
            </Box>
        </Box>
    );
};

export default AIChat;
