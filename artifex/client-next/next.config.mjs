/** @type {import('next').NextConfig} */
const nextConfig = {
  // Flask backend URL — no Express proxy needed
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001",
  },
  // Allow images from Flask backend
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
