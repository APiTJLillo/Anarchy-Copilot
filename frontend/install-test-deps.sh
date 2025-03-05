#!/bin/bash
cd "$(dirname "$0")"

echo "Installing testing dependencies..."
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest

echo "Installing Redux and testing dependencies..."
npm install @reduxjs/toolkit react-redux redux redux-thunk @types/redux-thunk

echo "Installing types for testing..."
npm install --save-dev @types/jest @types/testing-library__jest-dom

echo "Setting up Jest config..."
echo '{
  "testEnvironment": "jsdom",
  "transform": {
    "^.+\\.(ts|tsx)$": "ts-jest"
  },
  "setupFilesAfterEnv": [
    "@testing-library/jest-dom/extend-expect"
  ],
  "moduleNameMapper": {
    "\\.(css|less|sass|scss)$": "identity-obj-proxy"
  }
}' > jest.config.json

echo "Done! You can now run tests with 'npm test'"
