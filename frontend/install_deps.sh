#!/bin/bash
cd "$(dirname "$0")"

echo "Installing Redux and related dependencies..."
npm install @reduxjs/toolkit react-redux redux redux-thunk @types/redux-thunk

echo "Installing Material-UI dependencies..."
npm install @mui/material @mui/icons-material @emotion/react @emotion/styled

echo "Installing other dependencies..."
npm install axios@1.4.0

echo "Clearing npm cache..."
npm cache clean --force

echo "Done! Please restart your development server."
