// api/recommend.js
// Example proxy file for serverless deployment (Vercel, Netlify, etc.)

export default async function handler(req, res) {
    // Only allow POST requests
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // Get backend URL and secret from environment variables
    const BACKEND_URL = process.env.BACKEND_URL;
    const FRONTEND_PROXY_SECRET = process.env.FRONTEND_PROXY_SECRET;

    if (!BACKEND_URL || !FRONTEND_PROXY_SECRET) {
        return res.status(500).json({ error: 'Server configuration error' });
    }

    try {
        // Forward request to backend with secret header
        const backendResponse = await fetch(`${BACKEND_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'x-frontend-proxy-secret': FRONTEND_PROXY_SECRET
            },
            body: JSON.stringify(req.body)
        });

        if (!backendResponse.ok) {
            throw new Error(`Backend returned ${backendResponse.status}`);
        }

        const data = await backendResponse.json();
        
        // Return backend response unchanged
        return res.status(200).json(data);
        
    } catch (error) {
        console.error('Proxy error:', error);
        return res.status(500).json({ 
            error: 'Failed to process recommendation request' 
        });
    }
}
