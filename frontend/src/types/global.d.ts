declare module "*.css" {
    const content: { [className: string]: string };
    export default content;
}

declare module "*.scss" {
    const content: { [className: string]: string };
    export default content;
}

declare module "*.svg" {
    const content: string;
    export default content;
}

declare module "*.png" {
    const content: string;
    export default content;
}

declare module "*.jpg" {
    const content: string;
    export default content;
}

declare module "*.json" {
    const content: any;
    export default content;
}

// Module declarations for our app structure
declare module "components/*" {
    const component: any;
    export default component;
}

declare module "store/*" {
    const content: any;
    export default content;
}

declare module "api/*" {
    const content: any;
    export default content;
}

declare module "hooks/*" {
    const content: any;
    export default content;
}

declare module "contexts/*" {
    const content: any;
    export default content;
}

declare module "types/*" {
    const content: any;
    export default content;
}

declare module "__tests__/*" {
    const content: any;
    export default content;
}
