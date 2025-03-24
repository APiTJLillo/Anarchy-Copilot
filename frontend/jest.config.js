module.exports = {
    preset: 'ts-jest',
    testEnvironment: 'jsdom',
    setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
    moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
    moduleNameMapper: {
        // Add any module name mappings here if needed
        '^@/(.*)$': '<rootDir>/src/$1',
    },
    testMatch: [
        "**/__tests__/**/*.[jt]s?(x)",
        "**/?(*.)+(spec|test).[jt]s?(x)"
    ],
    globals: {
        'ts-jest': {
            tsconfig: '<rootDir>/tsconfig.json',
            diagnostics: false
        }
    },
    transform: {
        "^.+\\.(ts|tsx)$": "ts-jest"
    },
    testPathIgnorePatterns: [
        "/node_modules/",
        "/build/"
    ],
    collectCoverageFrom: [
        "src/**/*.{ts,tsx}",
        "!src/**/*.d.ts"
    ],
    coverageReporters: [
        "json",
        "lcov",
        "text",
        "clover"
    ],
    coverageDirectory: "coverage"
};
