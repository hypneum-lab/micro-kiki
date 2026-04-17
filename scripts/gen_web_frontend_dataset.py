#!/usr/bin/env python3
"""Generate web frontend training dataset.

Covers: React 19, Vite, Tailwind CSS, TypeScript for frontend,
state management, testing, performance, accessibility.

Target: ~2000 Q&A pairs.
Output: JSONL to stdout.

Usage::

    uv run python scripts/gen_web_frontend_dataset.py > /tmp/gen-web-frontend.jsonl
"""
from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def msg(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# 1. React 19
# ---------------------------------------------------------------------------

def gen_react19() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the key new features in React 19?",
        "React 19 introduces several major features:\n\n**1. Actions:**\nFunctions that handle async transitions, replacing manual `useState` + `useEffect` patterns for data mutations:\n```tsx\nfunction UpdateName() {\n  const [error, submitAction, isPending] = useActionState(\n    async (prev, formData) => {\n      const name = formData.get('name');\n      const error = await updateName(name);\n      if (error) return error;\n      redirect('/profile');\n      return null;\n    },\n    null\n  );\n\n  return (\n    <form action={submitAction}>\n      <input name=\"name\" />\n      <button disabled={isPending}>Update</button>\n      {error && <p>{error}</p>}\n    </form>\n  );\n}\n```\n\n**2. `use()` hook:**\nRead resources (promises, context) during render:\n```tsx\nfunction Comments({ commentsPromise }) {\n  const comments = use(commentsPromise); // suspends until resolved\n  return comments.map(c => <p key={c.id}>{c.text}</p>);\n}\n```\n\n**3. Server Components (RSC):**\nComponents that run only on the server, excluded from the client bundle. Can `await` directly.\n\n**4. Document Metadata:**\n`<title>`, `<meta>`, `<link>` can be rendered anywhere in the component tree — React hoists them to `<head>`.\n\n**5. `ref` as a prop:**\nNo more `forwardRef` — `ref` is now a regular prop on function components.\n\n**6. Improved error reporting:**\nDuplicate error logs removed, `onCaughtError` and `onUncaughtError` callbacks on `createRoot`."
    ))

    pairs.append(msg(
        "How does useActionState work in React 19?",
        "`useActionState` manages async form submissions with pending state and error handling.\n\n**Signature:**\n```tsx\nconst [state, formAction, isPending] = useActionState(actionFn, initialState, permalink?);\n```\n\n**Parameters:**\n- `actionFn(previousState, formData)`: Async function called on form submit\n- `initialState`: Initial state value\n- `permalink` (optional): URL for progressive enhancement with SSR\n\n**Returns:**\n- `state`: Current state (return value of the action)\n- `formAction`: Action to pass to `<form action={...}>`\n- `isPending`: Whether the action is in progress\n\n**Full example:**\n```tsx\nimport { useActionState } from 'react';\n\ntype State = { message: string } | null;\n\nasync function addTodo(prev: State, formData: FormData): Promise<State> {\n  const title = formData.get('title') as string;\n  if (!title.trim()) {\n    return { message: 'Title is required' };\n  }\n  try {\n    await fetch('/api/todos', {\n      method: 'POST',\n      body: JSON.stringify({ title }),\n      headers: { 'Content-Type': 'application/json' },\n    });\n    return { message: 'Todo added!' };\n  } catch {\n    return { message: 'Failed to add todo' };\n  }\n}\n\nexport function TodoForm() {\n  const [state, action, isPending] = useActionState(addTodo, null);\n\n  return (\n    <form action={action}>\n      <input name=\"title\" placeholder=\"New todo\" disabled={isPending} />\n      <button type=\"submit\" disabled={isPending}>\n        {isPending ? 'Adding...' : 'Add'}\n      </button>\n      {state?.message && <p>{state.message}</p>}\n    </form>\n  );\n}\n```\n\n**Key benefits over manual useState/useEffect:**\n- Automatic pending state management\n- Progressive enhancement (works without JS when using `permalink`)\n- Proper error boundary integration\n- No stale closure issues"
    ))

    pairs.append(msg(
        "How does the use() hook work in React 19?",
        "The `use()` hook reads the value of a resource (Promise or Context) during render.\n\n**Reading a Promise:**\n```tsx\nimport { use, Suspense } from 'react';\n\nfunction UserProfile({ userPromise }: { userPromise: Promise<User> }) {\n  const user = use(userPromise); // Suspends until resolved\n  return <h1>{user.name}</h1>;\n}\n\n// Parent wraps with Suspense\nfunction App() {\n  const userPromise = fetchUser(1); // Create promise outside render\n  return (\n    <Suspense fallback={<Spinner />}>\n      <UserProfile userPromise={userPromise} />\n    </Suspense>\n  );\n}\n```\n\n**Reading Context (replaces useContext):**\n```tsx\nimport { use } from 'react';\nimport { ThemeContext } from './ThemeContext';\n\nfunction Button() {\n  const theme = use(ThemeContext);\n  return <button className={theme.buttonClass}>Click</button>;\n}\n```\n\n**Key differences from useContext/await:**\n- `use()` can be called **conditionally** (inside if/else, loops) — unlike other hooks\n- It integrates with Suspense boundaries for loading states\n- For Promises: the component suspends, showing the nearest `<Suspense>` fallback\n- The Promise must be created outside the component render (not inside) to avoid re-creating it every render\n\n**Error handling:**\n```tsx\nimport { ErrorBoundary } from 'react-error-boundary';\n\n<ErrorBoundary fallback={<Error />}>\n  <Suspense fallback={<Loading />}>\n    <UserProfile userPromise={fetchUser(id)} />\n  </Suspense>\n</ErrorBoundary>\n```"
    ))

    pairs.append(msg(
        "How do React Server Components (RSC) work?",
        "**React Server Components (RSC)** render on the server and send serialized UI to the client. They never re-render on the client.\n\n**Server Component (default in RSC frameworks):**\n```tsx\n// app/users/page.tsx (Next.js App Router)\n// No 'use client' directive = Server Component\nimport { db } from '@/lib/db';\n\nexport default async function UsersPage() {\n  // Direct database access — runs on server only\n  const users = await db.query('SELECT * FROM users');\n\n  return (\n    <div>\n      <h1>Users ({users.length})</h1>\n      <ul>\n        {users.map(u => <li key={u.id}>{u.name}</li>)}\n      </ul>\n      <AddUserButton /> {/* Client Component */}\n    </div>\n  );\n}\n```\n\n**Client Component:**\n```tsx\n'use client';\nimport { useState } from 'react';\n\nexport function AddUserButton() {\n  const [isOpen, setIsOpen] = useState(false);\n  return (\n    <>\n      <button onClick={() => setIsOpen(true)}>Add User</button>\n      {isOpen && <UserForm />}\n    </>\n  );\n}\n```\n\n**Rules:**\n1. Server Components can import Client Components, but NOT vice versa\n2. Server Components can `await` (they're async by default)\n3. Server Components cannot use hooks (`useState`, `useEffect`, etc.)\n4. Server Components cannot use browser APIs (`window`, `document`)\n5. Client Components use `'use client'` directive at the top\n6. Props passed from Server to Client must be serializable (no functions, classes)\n\n**Benefits:**\n- Zero client-side JS for server components\n- Direct database/filesystem access\n- Automatic code splitting\n- Streaming SSR with Suspense"
    ))

    pairs.append(msg(
        "How does ref work as a prop in React 19 (no more forwardRef)?",
        "In React 19, `ref` is a regular prop on function components — `forwardRef` is no longer needed.\n\n**Before (React 18):**\n```tsx\nconst Input = forwardRef<HTMLInputElement, InputProps>((props, ref) => {\n  return <input ref={ref} {...props} />;\n});\n```\n\n**After (React 19):**\n```tsx\nfunction Input({ ref, ...props }: InputProps & { ref?: React.Ref<HTMLInputElement> }) {\n  return <input ref={ref} {...props} />;\n}\n\n// Usage — same as before\nfunction Form() {\n  const inputRef = useRef<HTMLInputElement>(null);\n  return <Input ref={inputRef} placeholder=\"Name\" />;\n}\n```\n\n**Ref cleanup functions** (new in React 19):\n```tsx\n<div ref={(node) => {\n  // Setup\n  const observer = new IntersectionObserver(callback);\n  observer.observe(node);\n\n  // Return cleanup function (like useEffect)\n  return () => observer.disconnect();\n}} />\n```\n\nThe cleanup function runs when the element is removed from the DOM, replacing the old pattern of checking for `null` in the ref callback.\n\n**Migration:** `forwardRef` still works but is deprecated. React 19 codemod handles the conversion automatically:\n```bash\nnpx @react-codemod/cli react-19 ./src\n```"
    ))

    pairs.append(msg(
        "What are React 19 Actions and how do they differ from React 18 patterns?",
        "**Actions** are async functions used in transitions. React 19 adds native support for async transitions.\n\n**React 18 pattern (manual):**\n```tsx\nfunction UpdateForm() {\n  const [name, setName] = useState('');\n  const [error, setError] = useState(null);\n  const [isPending, setIsPending] = useState(false);\n\n  const handleSubmit = async () => {\n    setIsPending(true);\n    setError(null);\n    try {\n      await updateName(name);\n    } catch (e) {\n      setError(e.message);\n    } finally {\n      setIsPending(false);\n    }\n  };\n\n  return (\n    <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>\n      <input value={name} onChange={e => setName(e.target.value)} />\n      <button disabled={isPending}>Update</button>\n      {error && <p>{error}</p>}\n    </form>\n  );\n}\n```\n\n**React 19 with Actions:**\n```tsx\nfunction UpdateForm() {\n  const [error, submitAction, isPending] = useActionState(\n    async (prev, formData) => {\n      const error = await updateName(formData.get('name'));\n      return error || null;\n    },\n    null\n  );\n\n  return (\n    <form action={submitAction}>\n      <input name=\"name\" />\n      <button disabled={isPending}>Update</button>\n      {error && <p>{error}</p>}\n    </form>\n  );\n}\n```\n\n**Key differences:**\n1. No manual `isPending` state — handled by the transition\n2. No `e.preventDefault()` — native `<form action>` integration\n3. Progressive enhancement — works without JavaScript\n4. Optimistic updates via `useOptimistic`:\n   ```tsx\n   const [optimisticName, setOptimisticName] = useOptimistic(name);\n   ```\n5. Error boundaries catch action errors automatically"
    ))

    pairs.append(msg(
        "How does useOptimistic work in React 19?",
        "`useOptimistic` shows a temporary optimistic state while an async action is pending.\n\n```tsx\nimport { useOptimistic, useActionState } from 'react';\n\ninterface Todo {\n  id: string;\n  text: string;\n  completed: boolean;\n}\n\nfunction TodoList({ todos }: { todos: Todo[] }) {\n  const [optimisticTodos, addOptimisticTodo] = useOptimistic(\n    todos,\n    // Reducer: current state + optimistic value → new state\n    (state, newTodo: Todo) => [...state, newTodo]\n  );\n\n  const [, formAction] = useActionState(\n    async (prev: null, formData: FormData) => {\n      const text = formData.get('text') as string;\n      const optimisticTodo: Todo = {\n        id: `temp-${Date.now()}`,\n        text,\n        completed: false,\n      };\n\n      // Show optimistic update immediately\n      addOptimisticTodo(optimisticTodo);\n\n      // Actually send to server\n      await createTodo(text);\n      return null;\n    },\n    null\n  );\n\n  return (\n    <>\n      <form action={formAction}>\n        <input name=\"text\" />\n        <button>Add Todo</button>\n      </form>\n      <ul>\n        {optimisticTodos.map(todo => (\n          <li key={todo.id} style={{\n            opacity: todo.id.startsWith('temp-') ? 0.5 : 1\n          }}>\n            {todo.text}\n          </li>\n        ))}\n      </ul>\n    </>\n  );\n}\n```\n\n**How it works:**\n1. `useOptimistic(currentState, updateFn)` returns `[optimisticState, addOptimistic]`\n2. When you call `addOptimistic(value)`, the reducer runs and returns the optimistic state\n3. The optimistic state is shown immediately\n4. When the action completes (or errors), React reverts to the real state from `currentState`\n5. If the action succeeds and `todos` prop updates, the optimistic state naturally resolves"
    ))

    pairs.append(msg(
        "How do you handle document metadata in React 19?",
        "React 19 natively supports rendering `<title>`, `<meta>`, and `<link>` tags anywhere in the component tree. React automatically hoists them to the `<head>`.\n\n```tsx\nfunction BlogPost({ post }: { post: Post }) {\n  return (\n    <article>\n      {/* These get hoisted to <head> automatically */}\n      <title>{post.title} - My Blog</title>\n      <meta name=\"description\" content={post.excerpt} />\n      <meta property=\"og:title\" content={post.title} />\n      <meta property=\"og:image\" content={post.coverImage} />\n      <link rel=\"canonical\" href={`https://myblog.com/posts/${post.slug}`} />\n\n      <h1>{post.title}</h1>\n      <p>{post.content}</p>\n    </article>\n  );\n}\n\n// Nested components can also set metadata\nfunction ProductPage({ product }: { product: Product }) {\n  return (\n    <>\n      <title>{product.name} | Shop</title>\n      <meta name=\"description\" content={product.description} />\n      <link\n        rel=\"stylesheet\"\n        href={`/themes/${product.theme}.css`}\n        precedence=\"default\"  // Controls ordering of stylesheets\n      />\n      <ProductDetails product={product} />\n    </>\n  );\n}\n```\n\n**Stylesheet support:**\n```tsx\n// React handles deduplication and ordering\n<link rel=\"stylesheet\" href=\"/base.css\" precedence=\"default\" />\n<link rel=\"stylesheet\" href=\"/theme.css\" precedence=\"high\" />\n```\n\nThe `precedence` prop controls CSS ordering in `<head>`. React deduplicates stylesheets and manages loading order.\n\n**Note:** This replaces libraries like `react-helmet` and `next/head` for most use cases."
    ))

    # Hooks deep dive
    pairs.append(msg(
        "What are the rules of React hooks and what problems do they prevent?",
        "**Rules of Hooks:**\n\n1. **Only call hooks at the top level** — not inside loops, conditions, or nested functions\n2. **Only call hooks from React function components or custom hooks** — not from regular JS functions\n\n**Why these rules exist:**\nReact relies on the **call order** of hooks to match state values between renders. Hooks don't have names — React identifies them by position.\n\n```tsx\n// BAD — conditional hook call\nfunction Component({ isLoggedIn }) {\n  if (isLoggedIn) {\n    const [user, setUser] = useState(null); // Position changes!\n  }\n  const [theme, setTheme] = useState('light');\n}\n\n// Render 1 (isLoggedIn=true):  Hook 1 = user, Hook 2 = theme\n// Render 2 (isLoggedIn=false): Hook 1 = theme (WRONG! Gets user's state)\n```\n\n**Correct pattern:**\n```tsx\nfunction Component({ isLoggedIn }) {\n  const [user, setUser] = useState(null);     // Always called\n  const [theme, setTheme] = useState('light'); // Always called\n\n  // Use the value conditionally, not the hook\n  const displayUser = isLoggedIn ? user : null;\n}\n```\n\n**ESLint enforcement:**\n```json\n{\n  \"plugins\": [\"react-hooks\"],\n  \"rules\": {\n    \"react-hooks/rules-of-hooks\": \"error\",\n    \"react-hooks/exhaustive-deps\": \"warn\"\n  }\n}\n```\n\n**Custom hooks** must start with `use` (convention + lint rule):\n```tsx\nfunction useWindowSize() {\n  const [size, setSize] = useState({ width: 0, height: 0 });\n  useEffect(() => {\n    const handler = () => setSize({\n      width: window.innerWidth,\n      height: window.innerHeight,\n    });\n    window.addEventListener('resize', handler);\n    handler();\n    return () => window.removeEventListener('resize', handler);\n  }, []);\n  return size;\n}\n```"
    ))

    pairs.append(msg(
        "How do you properly handle effects and cleanup in React?",
        "**useEffect** handles side effects — subscriptions, DOM manipulation, data fetching.\n\n**Anatomy of useEffect:**\n```tsx\nuseEffect(() => {\n  // Setup: runs after render\n  const subscription = api.subscribe(handler);\n\n  // Cleanup: runs before next effect and on unmount\n  return () => {\n    subscription.unsubscribe();\n  };\n}, [dependency1, dependency2]); // Dependency array\n```\n\n**Dependency array patterns:**\n```tsx\n// Runs once on mount (empty deps)\nuseEffect(() => {\n  initializeAnalytics();\n}, []);\n\n// Runs when userId changes\nuseEffect(() => {\n  fetchUser(userId).then(setUser);\n}, [userId]);\n\n// Runs on every render (no deps array — usually wrong)\nuseEffect(() => {\n  document.title = `Count: ${count}`;\n});\n```\n\n**Common cleanup patterns:**\n```tsx\n// Timer\nuseEffect(() => {\n  const id = setInterval(tick, 1000);\n  return () => clearInterval(id);\n}, []);\n\n// Event listener\nuseEffect(() => {\n  const handler = (e: KeyboardEvent) => {\n    if (e.key === 'Escape') onClose();\n  };\n  document.addEventListener('keydown', handler);\n  return () => document.removeEventListener('keydown', handler);\n}, [onClose]);\n\n// AbortController for fetch\nuseEffect(() => {\n  const controller = new AbortController();\n  fetch(`/api/users/${id}`, { signal: controller.signal })\n    .then(r => r.json())\n    .then(setUser)\n    .catch(e => {\n      if (e.name !== 'AbortError') setError(e);\n    });\n  return () => controller.abort();\n}, [id]);\n\n// WebSocket\nuseEffect(() => {\n  const ws = new WebSocket(url);\n  ws.onmessage = (e) => setMessages(prev => [...prev, JSON.parse(e.data)]);\n  return () => ws.close();\n}, [url]);\n```\n\n**React Strict Mode** runs effects twice in development to catch missing cleanup."
    ))

    return pairs


# ---------------------------------------------------------------------------
# 2. Vite Configuration
# ---------------------------------------------------------------------------

def gen_vite() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you set up a Vite project with React and TypeScript?",
        "```bash\nnpm create vite@latest my-app -- --template react-ts\ncd my-app\nnpm install\nnpm run dev\n```\n\n**Project structure:**\n```\nmy-app/\n├── index.html              # Entry point (Vite serves this)\n├── vite.config.ts          # Vite configuration\n├── tsconfig.json           # TypeScript config\n├── tsconfig.app.json       # App-specific TS config\n├── tsconfig.node.json      # Node (vite config) TS config\n├── package.json\n├── public/                 # Static assets (copied as-is)\n│   └── favicon.svg\n└── src/\n    ├── main.tsx            # React entry point\n    ├── App.tsx\n    ├── App.css\n    ├── index.css\n    └── vite-env.d.ts       # Vite type declarations\n```\n\n**Minimal vite.config.ts:**\n```typescript\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({\n  plugins: [react()],\n  server: {\n    port: 3000,\n    open: true,\n  },\n});\n```\n\n**Key scripts (package.json):**\n```json\n{\n  \"scripts\": {\n    \"dev\": \"vite\",\n    \"build\": \"tsc -b && vite build\",\n    \"preview\": \"vite preview\",\n    \"lint\": \"eslint .\"\n  }\n}\n```"
    ))

    pairs.append(msg(
        "How do you configure Vite proxy for API calls during development?",
        "Configure the `server.proxy` option in `vite.config.ts` to forward API requests to a backend server:\n\n```typescript\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({\n  plugins: [react()],\n  server: {\n    port: 3000,\n    proxy: {\n      // Simple proxy: /api/* → http://localhost:8000/api/*\n      '/api': {\n        target: 'http://localhost:8000',\n        changeOrigin: true,\n      },\n\n      // Rewrite path: /api/* → http://localhost:8000/*\n      '/api': {\n        target: 'http://localhost:8000',\n        changeOrigin: true,\n        rewrite: (path) => path.replace(/^\\/api/, ''),\n      },\n\n      // WebSocket proxy\n      '/ws': {\n        target: 'ws://localhost:8000',\n        ws: true,\n      },\n\n      // Multiple backends\n      '/auth': {\n        target: 'http://localhost:8080',\n        changeOrigin: true,\n      },\n\n      // With custom headers\n      '/external': {\n        target: 'https://api.example.com',\n        changeOrigin: true,\n        headers: {\n          'X-Custom-Header': 'value',\n        },\n        secure: false, // Accept self-signed certs\n      },\n    },\n  },\n});\n```\n\n**In your React code**, just use relative paths:\n```typescript\n// No need for CORS or full URLs during development\nconst response = await fetch('/api/users');\n```\n\nThe proxy only applies during `vite dev`. For production, configure your reverse proxy (nginx, Caddy) or API gateway."
    ))

    pairs.append(msg(
        "How do you optimize a Vite build for production?",
        "```typescript\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({\n  plugins: [react()],\n  build: {\n    // Target modern browsers (smaller output)\n    target: 'es2022',\n\n    // Chunk splitting strategy\n    rollupOptions: {\n      output: {\n        manualChunks: {\n          // Vendor splitting\n          'react-vendor': ['react', 'react-dom'],\n          'router': ['react-router-dom'],\n          'ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],\n        },\n      },\n    },\n\n    // Enable CSS code splitting\n    cssCodeSplit: true,\n\n    // Source maps for debugging (disable for smaller builds)\n    sourcemap: false,\n\n    // Chunk size warning limit (kB)\n    chunkSizeWarningLimit: 500,\n\n    // Minification (default: esbuild, faster)\n    // Use terser for maximum compression:\n    // minify: 'terser',\n    // terserOptions: { compress: { drop_console: true } },\n  },\n\n  // Environment-specific\n  define: {\n    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),\n  },\n\n  // CSS processing\n  css: {\n    devSourcemap: true,\n    modules: {\n      localsConvention: 'camelCaseOnly',\n    },\n  },\n});\n```\n\n**Analyzing bundle size:**\n```bash\nnpm install -D rollup-plugin-visualizer\n```\n```typescript\nimport { visualizer } from 'rollup-plugin-visualizer';\n\nplugins: [\n  react(),\n  visualizer({ open: true, gzipSize: true }),\n],\n```\n\n**Key optimizations:**\n- Dynamic imports for route-based code splitting\n- `target: 'es2022'` avoids unnecessary polyfills\n- Manual chunks prevent vendor code re-downloading on app updates\n- CSS code splitting keeps per-route CSS small\n- `drop_console` in production removes debug logs"
    ))

    pairs.append(msg(
        "How do you configure environment variables in Vite?",
        "Vite uses `.env` files with a `VITE_` prefix for client-exposed variables.\n\n**File hierarchy (later overrides earlier):**\n```\n.env                 # Always loaded\n.env.local           # Always loaded, gitignored\n.env.development     # Only during `vite dev`\n.env.production      # Only during `vite build`\n.env.staging         # Custom mode: `vite build --mode staging`\n```\n\n**Example .env:**\n```\n# Exposed to client (VITE_ prefix required)\nVITE_API_URL=http://localhost:8000\nVITE_APP_TITLE=My App\nVITE_FEATURE_FLAG_NEW_UI=true\n\n# NOT exposed to client (no VITE_ prefix)\nDATABASE_URL=postgres://localhost/mydb\nSECRET_KEY=abc123\n```\n\n**Accessing in code:**\n```typescript\n// In .ts/.tsx files\nconst apiUrl = import.meta.env.VITE_API_URL;\nconst isDev = import.meta.env.DEV;      // boolean\nconst isProd = import.meta.env.PROD;    // boolean\nconst mode = import.meta.env.MODE;      // 'development' | 'production'\n\n// Type safety — create src/vite-env.d.ts:\ninterface ImportMetaEnv {\n  readonly VITE_API_URL: string;\n  readonly VITE_APP_TITLE: string;\n  readonly VITE_FEATURE_FLAG_NEW_UI: string;\n}\n\ninterface ImportMeta {\n  readonly env: ImportMetaEnv;\n}\n```\n\n**In vite.config.ts** (runs in Node, not browser):\n```typescript\nimport { defineConfig, loadEnv } from 'vite';\n\nexport default defineConfig(({ mode }) => {\n  const env = loadEnv(mode, process.cwd(), '');\n  return {\n    define: {\n      __API_URL__: JSON.stringify(env.VITE_API_URL),\n    },\n  };\n});\n```\n\n**Security:** Never put secrets in `VITE_` variables — they are embedded in the client bundle."
    ))

    pairs.append(msg(
        "How do you add path aliases in Vite with TypeScript?",
        "Configure both `vite.config.ts` and `tsconfig.json` to use path aliases.\n\n**vite.config.ts:**\n```typescript\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\nimport path from 'path';\n\nexport default defineConfig({\n  plugins: [react()],\n  resolve: {\n    alias: {\n      '@': path.resolve(__dirname, './src'),\n      '@components': path.resolve(__dirname, './src/components'),\n      '@hooks': path.resolve(__dirname, './src/hooks'),\n      '@utils': path.resolve(__dirname, './src/utils'),\n      '@api': path.resolve(__dirname, './src/api'),\n    },\n  },\n});\n```\n\n**tsconfig.json (or tsconfig.app.json):**\n```json\n{\n  \"compilerOptions\": {\n    \"baseUrl\": \".\",\n    \"paths\": {\n      \"@/*\": [\"src/*\"],\n      \"@components/*\": [\"src/components/*\"],\n      \"@hooks/*\": [\"src/hooks/*\"],\n      \"@utils/*\": [\"src/utils/*\"],\n      \"@api/*\": [\"src/api/*\"]\n    }\n  }\n}\n```\n\n**Usage in code:**\n```typescript\n// Instead of: import { Button } from '../../../components/Button'\nimport { Button } from '@components/Button';\nimport { useAuth } from '@hooks/useAuth';\nimport { formatDate } from '@utils/date';\nimport { api } from '@api/client';\n```\n\n**Alternative — vite-tsconfig-paths plugin** (auto-syncs from tsconfig):\n```bash\nnpm install -D vite-tsconfig-paths\n```\n```typescript\nimport tsconfigPaths from 'vite-tsconfig-paths';\n\nexport default defineConfig({\n  plugins: [react(), tsconfigPaths()],\n});\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 3. Tailwind CSS
# ---------------------------------------------------------------------------

def gen_tailwind() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you set up Tailwind CSS v4 with Vite?",
        "Tailwind CSS v4 uses a CSS-first configuration approach.\n\n**Installation:**\n```bash\nnpm install tailwindcss @tailwindcss/vite\n```\n\n**vite.config.ts:**\n```typescript\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\nimport tailwindcss from '@tailwindcss/vite';\n\nexport default defineConfig({\n  plugins: [react(), tailwindcss()],\n});\n```\n\n**src/index.css:**\n```css\n@import \"tailwindcss\";\n\n/* Custom theme configuration (v4 style) */\n@theme {\n  --color-primary: #3b82f6;\n  --color-secondary: #8b5cf6;\n  --font-sans: 'Inter', sans-serif;\n  --breakpoint-xs: 475px;\n}\n\n/* Custom utilities */\n@utility text-balance {\n  text-wrap: balance;\n}\n```\n\n**v4 changes from v3:**\n- No `tailwind.config.js` — configuration is in CSS via `@theme`\n- No `@tailwind base/components/utilities` — just `@import \"tailwindcss\"`\n- CSS-native nesting and variables\n- Automatic content detection (no `content` config needed)\n- New `@utility` directive for custom utilities\n- Built-in container queries, `@starting-style`, etc.\n\n**Usage in React:**\n```tsx\nfunction Button({ children, variant = 'primary' }) {\n  return (\n    <button className={`\n      px-4 py-2 rounded-lg font-medium transition-colors\n      ${variant === 'primary'\n        ? 'bg-primary text-white hover:bg-primary/90'\n        : 'bg-gray-100 text-gray-900 hover:bg-gray-200'\n      }\n    `}>\n      {children}\n    </button>\n  );\n}\n```"
    ))

    pairs.append(msg(
        "How do you handle responsive design with Tailwind CSS?",
        "Tailwind uses **mobile-first breakpoint prefixes**. Unprefixed classes apply to all screens, prefixed classes apply at that breakpoint and above.\n\n**Default breakpoints:**\n```\nsm:  640px\nmd:  768px\nlg:  1024px\nxl:  1280px\n2xl: 1536px\n```\n\n**Examples:**\n```tsx\n// Stack on mobile, side-by-side on desktop\n<div className=\"flex flex-col md:flex-row gap-4\">\n  <aside className=\"w-full md:w-64\">Sidebar</aside>\n  <main className=\"flex-1\">Content</main>\n</div>\n\n// Responsive grid\n<div className=\"grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6\">\n  {items.map(item => <Card key={item.id} item={item} />)}\n</div>\n\n// Responsive typography\n<h1 className=\"text-2xl sm:text-3xl lg:text-5xl font-bold\">\n  Responsive Heading\n</h1>\n\n// Hide/show at breakpoints\n<nav className=\"hidden md:block\">Desktop Nav</nav>\n<button className=\"md:hidden\">Mobile Menu</button>\n\n// Responsive padding\n<section className=\"px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-16\">\n  Content\n</section>\n```\n\n**Container queries (v4):**\n```tsx\n<div className=\"@container\">\n  <div className=\"@sm:flex @sm:gap-4\">\n    {/* Responds to container width, not viewport */}\n    <img className=\"w-full @sm:w-48\" />\n    <p>Text content</p>\n  </div>\n</div>\n```\n\n**Custom breakpoints (v4):**\n```css\n@theme {\n  --breakpoint-xs: 475px;\n  --breakpoint-3xl: 1920px;\n}\n```"
    ))

    pairs.append(msg(
        "How do you implement dark mode with Tailwind CSS?",
        "Tailwind supports dark mode with the `dark:` variant.\n\n**Strategy 1: System preference (default in v4):**\n```css\n/* Uses prefers-color-scheme media query automatically */\n```\n```tsx\n<div className=\"bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100\">\n  <h1 className=\"text-black dark:text-white\">Hello</h1>\n  <button className=\"bg-blue-500 dark:bg-blue-400 text-white\">\n    Click me\n  </button>\n</div>\n```\n\n**Strategy 2: Manual toggle (class-based):**\n```css\n/* In v4 CSS config */\n@custom-variant dark (&:where(.dark, .dark *));\n```\n\n```tsx\nimport { useState, useEffect } from 'react';\n\nfunction ThemeToggle() {\n  const [dark, setDark] = useState(() =>\n    localStorage.getItem('theme') === 'dark' ||\n    (!localStorage.getItem('theme') &&\n      window.matchMedia('(prefers-color-scheme: dark)').matches)\n  );\n\n  useEffect(() => {\n    document.documentElement.classList.toggle('dark', dark);\n    localStorage.setItem('theme', dark ? 'dark' : 'light');\n  }, [dark]);\n\n  return (\n    <button onClick={() => setDark(!dark)}\n            className=\"p-2 rounded-lg bg-gray-200 dark:bg-gray-700\">\n      {dark ? 'Light' : 'Dark'}\n    </button>\n  );\n}\n```\n\n**Design patterns:**\n```tsx\n// Card component with dark mode\n<div className=\"\n  bg-white dark:bg-gray-800\n  border border-gray-200 dark:border-gray-700\n  shadow-sm dark:shadow-gray-900/20\n  rounded-xl p-6\n\">\n  <h3 className=\"text-gray-900 dark:text-gray-100\">Title</h3>\n  <p className=\"text-gray-600 dark:text-gray-400\">Description</p>\n</div>\n```\n\n**Tip:** Use CSS custom properties for systematic dark mode:\n```css\n@theme {\n  --color-surface: #ffffff;\n  --color-text: #111827;\n}\n\n@media (prefers-color-scheme: dark) {\n  @theme {\n    --color-surface: #1f2937;\n    --color-text: #f9fafb;\n  }\n}\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 4. TypeScript for Frontend
# ---------------------------------------------------------------------------

def gen_typescript_frontend() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you type React component props with TypeScript?",
        "**Basic props typing:**\n```tsx\n// Interface for props\ninterface ButtonProps {\n  label: string;\n  onClick: () => void;\n  variant?: 'primary' | 'secondary' | 'danger';  // optional with union\n  size?: 'sm' | 'md' | 'lg';\n  disabled?: boolean;\n  icon?: React.ReactNode;  // any renderable content\n  className?: string;\n}\n\nfunction Button({ label, onClick, variant = 'primary', size = 'md',\n                   disabled = false, icon, className }: ButtonProps) {\n  return (\n    <button onClick={onClick} disabled={disabled} className={className}>\n      {icon && <span>{icon}</span>}\n      {label}\n    </button>\n  );\n}\n```\n\n**Children props:**\n```tsx\n// Explicit children\ninterface CardProps {\n  title: string;\n  children: React.ReactNode;  // anything renderable\n}\n\n// PropsWithChildren helper\nimport { PropsWithChildren } from 'react';\nfunction Card({ title, children }: PropsWithChildren<{ title: string }>) {\n  return <div><h2>{title}</h2>{children}</div>;\n}\n```\n\n**Extending HTML element props:**\n```tsx\n// Button that accepts all native button attributes\ninterface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {\n  variant?: 'primary' | 'secondary';\n}\n\nfunction Button({ variant = 'primary', children, ...rest }: ButtonProps) {\n  return <button {...rest}>{children}</button>;\n}\n\n// Input wrapper\ninterface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {\n  label: string;\n  error?: string;\n}\n\nfunction Input({ label, error, ...inputProps }: InputProps) {\n  return (\n    <div>\n      <label>{label}</label>\n      <input {...inputProps} />\n      {error && <span className=\"text-red-500\">{error}</span>}\n    </div>\n  );\n}\n```\n\n**Generic components:**\n```tsx\ninterface ListProps<T> {\n  items: T[];\n  renderItem: (item: T) => React.ReactNode;\n  keyExtractor: (item: T) => string;\n}\n\nfunction List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {\n  return <ul>{items.map(item => <li key={keyExtractor(item)}>{renderItem(item)}</li>)}</ul>;\n}\n\n// Usage — T is inferred\n<List\n  items={users}\n  renderItem={(user) => <span>{user.name}</span>}\n  keyExtractor={(user) => user.id}\n/>\n```"
    ))

    pairs.append(msg(
        "How do you type React hooks with TypeScript?",
        "**useState:**\n```tsx\n// Inferred type\nconst [count, setCount] = useState(0); // number\nconst [name, setName] = useState('');  // string\n\n// Explicit type (needed for complex types or null initial)\nconst [user, setUser] = useState<User | null>(null);\nconst [items, setItems] = useState<string[]>([]);\n\ninterface User {\n  id: string;\n  name: string;\n  email: string;\n}\n```\n\n**useRef:**\n```tsx\n// DOM element ref\nconst inputRef = useRef<HTMLInputElement>(null);\n// Later: inputRef.current?.focus();\n\n// Mutable value ref (no null)\nconst intervalRef = useRef<number>(0);\n// Later: intervalRef.current = window.setInterval(...);\n```\n\n**useReducer:**\n```tsx\ntype Action =\n  | { type: 'INCREMENT' }\n  | { type: 'DECREMENT' }\n  | { type: 'SET'; payload: number };\n\ninterface State {\n  count: number;\n}\n\nfunction reducer(state: State, action: Action): State {\n  switch (action.type) {\n    case 'INCREMENT': return { count: state.count + 1 };\n    case 'DECREMENT': return { count: state.count - 1 };\n    case 'SET': return { count: action.payload };\n  }\n}\n\nconst [state, dispatch] = useReducer(reducer, { count: 0 });\ndispatch({ type: 'SET', payload: 42 });\n```\n\n**useCallback and useMemo:**\n```tsx\nconst handleClick = useCallback((id: string) => {\n  deleteItem(id);\n}, [deleteItem]);\n// Type: (id: string) => void\n\nconst sortedItems = useMemo((): Item[] => {\n  return items.sort((a, b) => a.name.localeCompare(b.name));\n}, [items]);\n```\n\n**useContext:**\n```tsx\ninterface AuthContext {\n  user: User | null;\n  login: (email: string, password: string) => Promise<void>;\n  logout: () => void;\n}\n\nconst AuthContext = createContext<AuthContext | null>(null);\n\nfunction useAuth(): AuthContext {\n  const ctx = useContext(AuthContext);\n  if (!ctx) throw new Error('useAuth must be used within AuthProvider');\n  return ctx;\n}\n```"
    ))

    pairs.append(msg(
        "How do you use TypeScript generics effectively in React?",
        "**Generic components:**\n```tsx\n// Select component that preserves option type\ninterface SelectProps<T> {\n  options: T[];\n  value: T;\n  onChange: (value: T) => void;\n  getLabel: (option: T) => string;\n  getValue: (option: T) => string;\n}\n\nfunction Select<T>({ options, value, onChange, getLabel, getValue }: SelectProps<T>) {\n  return (\n    <select\n      value={getValue(value)}\n      onChange={(e) => {\n        const selected = options.find(o => getValue(o) === e.target.value);\n        if (selected) onChange(selected);\n      }}\n    >\n      {options.map(option => (\n        <option key={getValue(option)} value={getValue(option)}>\n          {getLabel(option)}\n        </option>\n      ))}\n    </select>\n  );\n}\n\n// Usage — type-safe\ninterface Country { code: string; name: string; }\n<Select<Country>\n  options={countries}\n  value={selected}\n  onChange={setSelected}  // (value: Country) => void\n  getLabel={c => c.name}\n  getValue={c => c.code}\n/>\n```\n\n**Generic hooks:**\n```tsx\nfunction useFetch<T>(url: string): {\n  data: T | null;\n  error: Error | null;\n  loading: boolean;\n} {\n  const [data, setData] = useState<T | null>(null);\n  const [error, setError] = useState<Error | null>(null);\n  const [loading, setLoading] = useState(true);\n\n  useEffect(() => {\n    const controller = new AbortController();\n    fetch(url, { signal: controller.signal })\n      .then(r => r.json())\n      .then((d: T) => { setData(d); setLoading(false); })\n      .catch(e => { if (e.name !== 'AbortError') { setError(e); setLoading(false); }});\n    return () => controller.abort();\n  }, [url]);\n\n  return { data, error, loading };\n}\n\n// Type-safe usage\nconst { data: users } = useFetch<User[]>('/api/users');\n// users is User[] | null\n```\n\n**Constrained generics:**\n```tsx\ninterface HasId { id: string | number; }\n\nfunction useSelection<T extends HasId>(items: T[]) {\n  const [selectedIds, setSelectedIds] = useState<Set<T['id']>>(new Set());\n  const toggle = (item: T) => {\n    setSelectedIds(prev => {\n      const next = new Set(prev);\n      next.has(item.id) ? next.delete(item.id) : next.add(item.id);\n      return next;\n    });\n  };\n  return { selectedIds, toggle };\n}\n```"
    ))

    pairs.append(msg(
        "What are discriminated unions in TypeScript and how do you use them in React?",
        "**Discriminated unions** use a shared literal property (discriminant) to narrow types.\n\n```tsx\n// API response type\ntype ApiResult<T> =\n  | { status: 'loading' }\n  | { status: 'success'; data: T }\n  | { status: 'error'; error: string };\n\nfunction UserProfile({ result }: { result: ApiResult<User> }) {\n  switch (result.status) {\n    case 'loading':\n      return <Spinner />;\n    case 'error':\n      return <Alert>{result.error}</Alert>;  // TypeScript knows error exists\n    case 'success':\n      return <h1>{result.data.name}</h1>;    // TypeScript knows data exists\n  }\n}\n```\n\n**Component variants:**\n```tsx\ntype ButtonProps =\n  | { variant: 'link'; href: string; onClick?: never }\n  | { variant: 'button'; onClick: () => void; href?: never }\n  | { variant: 'submit'; onClick?: never; href?: never };\n\nfunction ActionButton(props: ButtonProps & { children: React.ReactNode }) {\n  switch (props.variant) {\n    case 'link':\n      return <a href={props.href}>{props.children}</a>;\n    case 'button':\n      return <button onClick={props.onClick}>{props.children}</button>;\n    case 'submit':\n      return <button type=\"submit\">{props.children}</button>;\n  }\n}\n\n// Type-safe usage\n<ActionButton variant=\"link\" href=\"/about\">About</ActionButton>\n<ActionButton variant=\"button\" onClick={handleClick}>Click</ActionButton>\n// Error: <ActionButton variant=\"link\" onClick={fn}>  // href required for link\n```\n\n**Reducer actions:**\n```tsx\ntype Action =\n  | { type: 'ADD_TODO'; payload: { text: string } }\n  | { type: 'TOGGLE_TODO'; payload: { id: string } }\n  | { type: 'DELETE_TODO'; payload: { id: string } }\n  | { type: 'SET_FILTER'; payload: 'all' | 'active' | 'completed' };\n\nfunction reducer(state: State, action: Action): State {\n  switch (action.type) {\n    case 'ADD_TODO':\n      // action.payload is { text: string }\n      return { ...state, todos: [...state.todos, { id: crypto.randomUUID(), text: action.payload.text, completed: false }] };\n    case 'TOGGLE_TODO':\n      // action.payload is { id: string }\n      return { ...state, todos: state.todos.map(t => t.id === action.payload.id ? { ...t, completed: !t.completed } : t) };\n    // TypeScript exhaustiveness check\n  }\n}\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 5. State Management
# ---------------------------------------------------------------------------

def gen_state_management() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How does Zustand work for React state management?",
        "**Zustand** is a minimal state management library with a simple API.\n\n**Installation:**\n```bash\nnpm install zustand\n```\n\n**Creating a store:**\n```typescript\nimport { create } from 'zustand';\nimport { devtools, persist } from 'zustand/middleware';\n\ninterface TodoStore {\n  todos: Todo[];\n  filter: 'all' | 'active' | 'completed';\n  // Actions\n  addTodo: (text: string) => void;\n  toggleTodo: (id: string) => void;\n  deleteTodo: (id: string) => void;\n  setFilter: (filter: 'all' | 'active' | 'completed') => void;\n}\n\nconst useTodoStore = create<TodoStore>()(\n  devtools(\n    persist(\n      (set, get) => ({\n        todos: [],\n        filter: 'all',\n\n        addTodo: (text) => set((state) => ({\n          todos: [...state.todos, { id: crypto.randomUUID(), text, completed: false }],\n        })),\n\n        toggleTodo: (id) => set((state) => ({\n          todos: state.todos.map(t =>\n            t.id === id ? { ...t, completed: !t.completed } : t\n          ),\n        })),\n\n        deleteTodo: (id) => set((state) => ({\n          todos: state.todos.filter(t => t.id !== id),\n        })),\n\n        setFilter: (filter) => set({ filter }),\n      }),\n      { name: 'todo-storage' } // localStorage key\n    )\n  )\n);\n```\n\n**Using in components:**\n```tsx\nfunction TodoList() {\n  // Only re-renders when todos or filter changes\n  const todos = useTodoStore(s => s.todos);\n  const filter = useTodoStore(s => s.filter);\n  const toggleTodo = useTodoStore(s => s.toggleTodo);\n\n  const filtered = todos.filter(t => {\n    if (filter === 'active') return !t.completed;\n    if (filter === 'completed') return t.completed;\n    return true;\n  });\n\n  return (\n    <ul>\n      {filtered.map(t => (\n        <li key={t.id} onClick={() => toggleTodo(t.id)}>\n          {t.text}\n        </li>\n      ))}\n    </ul>\n  );\n}\n```\n\n**Key advantages:**\n- No Provider needed\n- Automatic render optimization (selector-based subscriptions)\n- Middleware: devtools, persist, immer, subscribeWithSelector\n- Works outside React (`useTodoStore.getState()`, `useTodoStore.setState()`)"
    ))

    pairs.append(msg(
        "How does Jotai compare to Zustand for state management?",
        "**Jotai** is an atomic state management library — state is composed of independent atoms.\n\n```typescript\nimport { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';\nimport { atomWithStorage } from 'jotai/utils';\n\n// Primitive atoms\nconst countAtom = atom(0);\nconst nameAtom = atom('');\nconst darkModeAtom = atomWithStorage('darkMode', false); // persisted\n\n// Derived atom (computed)\nconst doubleCountAtom = atom((get) => get(countAtom) * 2);\n\n// Writable derived atom\nconst uppercaseNameAtom = atom(\n  (get) => get(nameAtom).toUpperCase(),  // read\n  (get, set, newName: string) => {        // write\n    set(nameAtom, newName.trim());\n  }\n);\n\n// Async atom\nconst userAtom = atom(async () => {\n  const res = await fetch('/api/user');\n  return res.json();\n});\n\n// Usage in components\nfunction Counter() {\n  const [count, setCount] = useAtom(countAtom);\n  const doubleCount = useAtomValue(doubleCountAtom); // read-only\n\n  return (\n    <div>\n      <p>Count: {count}, Double: {doubleCount}</p>\n      <button onClick={() => setCount(c => c + 1)}>+1</button>\n    </div>\n  );\n}\n\n// Write-only (no re-render on atom change)\nfunction ResetButton() {\n  const setCount = useSetAtom(countAtom);\n  return <button onClick={() => setCount(0)}>Reset</button>;\n}\n```\n\n**Zustand vs Jotai:**\n| Feature | Zustand | Jotai |\n|---------|---------|-------|\n| Mental model | Single store (like Redux) | Atoms (like Recoil) |\n| State shape | One object with actions | Many independent atoms |\n| Re-renders | Selector-based | Atom-based (automatic) |\n| Provider | Not needed | Optional (for isolation) |\n| Best for | Shared global state | Fine-grained reactive state |\n| Async | Manual in actions | Built-in async atoms |\n| DevTools | Via middleware | Via jotai-devtools |\n\n**Use Zustand** for: centralized stores (auth, cart, settings).\n**Use Jotai** for: fine-grained state that many independent components share."
    ))

    pairs.append(msg(
        "When should you use React Context vs a state management library?",
        "**React Context** is for **infrequently-changing values** that many components need. It's NOT optimized for frequent updates.\n\n**Good for Context:**\n- Theme (dark/light)\n- Locale/language\n- Auth state (user object, login/logout)\n- Feature flags\n- App-level configuration\n\n**Bad for Context (use a library instead):**\n- Form state with many fields\n- Lists with frequent updates\n- Real-time data (WebSocket, polling)\n- Shopping cart items\n- Any state that changes multiple times per second\n\n**Why Context has performance issues:**\n```tsx\nconst AppContext = createContext({ user: null, theme: 'light', notifications: [] });\n\n// Problem: ANY change to the context re-renders ALL consumers\n// If notifications update every second, every theme consumer re-renders\n```\n\n**Mitigation pattern — split contexts:**\n```tsx\n// Split into separate contexts by update frequency\nconst ThemeContext = createContext('light');        // rarely changes\nconst AuthContext = createContext<User | null>(null); // rarely changes\nconst NotifContext = createContext<Notification[]>([]); // frequent\n\nfunction Providers({ children }) {\n  const [theme, setTheme] = useState('light');\n  const [user, setUser] = useState(null);\n  const [notifs, setNotifs] = useState([]);\n\n  return (\n    <ThemeContext.Provider value={theme}>\n      <AuthContext.Provider value={user}>\n        <NotifContext.Provider value={notifs}>\n          {children}\n        </NotifContext.Provider>\n      </AuthContext.Provider>\n    </ThemeContext.Provider>\n  );\n}\n```\n\n**Decision guide:**\n- < 5 consumers, rare updates → Context\n- Many consumers, frequent updates → Zustand/Jotai\n- Complex derived state → Zustand/Jotai\n- Need persistence → Zustand (persist middleware) / Jotai (atomWithStorage)"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 6. Testing
# ---------------------------------------------------------------------------

def gen_testing() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you set up Vitest for a React project?",
        "```bash\nnpm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom\n```\n\n**vite.config.ts:**\n```typescript\n/// <reference types=\"vitest/config\" />\nimport { defineConfig } from 'vite';\nimport react from '@vitejs/plugin-react';\n\nexport default defineConfig({\n  plugins: [react()],\n  test: {\n    globals: true,          // no need to import describe/it/expect\n    environment: 'jsdom',   // browser-like environment\n    setupFiles: './src/test/setup.ts',\n    css: true,              // process CSS imports\n    coverage: {\n      provider: 'v8',\n      reporter: ['text', 'html'],\n      include: ['src/**/*.{ts,tsx}'],\n      exclude: ['src/test/**', '**/*.d.ts'],\n    },\n  },\n});\n```\n\n**src/test/setup.ts:**\n```typescript\nimport '@testing-library/jest-dom/vitest';\n\n// Optional: global mocks\nObject.defineProperty(window, 'matchMedia', {\n  writable: true,\n  value: vi.fn().mockImplementation(query => ({\n    matches: false,\n    media: query,\n    addListener: vi.fn(),\n    removeListener: vi.fn(),\n  })),\n});\n```\n\n**Example test:**\n```tsx\nimport { render, screen } from '@testing-library/react';\nimport userEvent from '@testing-library/user-event';\nimport { Counter } from './Counter';\n\ndescribe('Counter', () => {\n  it('increments count on click', async () => {\n    const user = userEvent.setup();\n    render(<Counter />);\n\n    expect(screen.getByText('Count: 0')).toBeInTheDocument();\n\n    await user.click(screen.getByRole('button', { name: /increment/i }));\n\n    expect(screen.getByText('Count: 1')).toBeInTheDocument();\n  });\n});\n```\n\n**Scripts:**\n```json\n{\n  \"scripts\": {\n    \"test\": \"vitest\",\n    \"test:run\": \"vitest run\",\n    \"test:coverage\": \"vitest run --coverage\"\n  }\n}\n```"
    ))

    pairs.append(msg(
        "How do you test async operations and API calls in React components?",
        "Use **MSW** (Mock Service Worker) for API mocking and `waitFor`/`findBy` for async assertions.\n\n**Setup MSW:**\n```typescript\n// src/test/handlers.ts\nimport { http, HttpResponse } from 'msw';\n\nexport const handlers = [\n  http.get('/api/users', () => {\n    return HttpResponse.json([\n      { id: '1', name: 'Alice' },\n      { id: '2', name: 'Bob' },\n    ]);\n  }),\n\n  http.post('/api/users', async ({ request }) => {\n    const body = await request.json();\n    return HttpResponse.json({ id: '3', ...body }, { status: 201 });\n  }),\n\n  http.get('/api/users/:id', ({ params }) => {\n    if (params.id === '999') {\n      return HttpResponse.json(\n        { message: 'Not found' },\n        { status: 404 }\n      );\n    }\n    return HttpResponse.json({ id: params.id, name: 'Alice' });\n  }),\n];\n\n// src/test/server.ts\nimport { setupServer } from 'msw/node';\nimport { handlers } from './handlers';\nexport const server = setupServer(...handlers);\n\n// src/test/setup.ts\nimport { server } from './server';\nbeforeAll(() => server.listen());\nafterEach(() => server.resetHandlers());\nafterAll(() => server.close());\n```\n\n**Testing a component that fetches data:**\n```tsx\nimport { render, screen, waitFor } from '@testing-library/react';\nimport userEvent from '@testing-library/user-event';\nimport { UserList } from './UserList';\nimport { server } from '../test/server';\nimport { http, HttpResponse } from 'msw';\n\ndescribe('UserList', () => {\n  it('displays users after loading', async () => {\n    render(<UserList />);\n\n    // Initially shows loading\n    expect(screen.getByText(/loading/i)).toBeInTheDocument();\n\n    // Wait for data to appear\n    expect(await screen.findByText('Alice')).toBeInTheDocument();\n    expect(screen.getByText('Bob')).toBeInTheDocument();\n    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();\n  });\n\n  it('shows error on API failure', async () => {\n    // Override handler for this test\n    server.use(\n      http.get('/api/users', () => {\n        return HttpResponse.json(null, { status: 500 });\n      })\n    );\n\n    render(<UserList />);\n\n    expect(await screen.findByText(/error/i)).toBeInTheDocument();\n  });\n});\n```"
    ))

    pairs.append(msg(
        "How do you write Playwright end-to-end tests for a React app?",
        "```bash\nnpm init playwright@latest\n```\n\n**playwright.config.ts:**\n```typescript\nimport { defineConfig, devices } from '@playwright/test';\n\nexport default defineConfig({\n  testDir: './e2e',\n  fullyParallel: true,\n  retries: process.env.CI ? 2 : 0,\n  workers: process.env.CI ? 1 : undefined,\n  reporter: 'html',\n  use: {\n    baseURL: 'http://localhost:3000',\n    trace: 'on-first-retry',\n    screenshot: 'only-on-failure',\n  },\n  projects: [\n    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },\n    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },\n    { name: 'mobile', use: { ...devices['iPhone 14'] } },\n  ],\n  webServer: {\n    command: 'npm run dev',\n    url: 'http://localhost:3000',\n    reuseExistingServer: !process.env.CI,\n  },\n});\n```\n\n**Example E2E test:**\n```typescript\nimport { test, expect } from '@playwright/test';\n\ntest.describe('Todo App', () => {\n  test.beforeEach(async ({ page }) => {\n    await page.goto('/');\n  });\n\n  test('adds a new todo', async ({ page }) => {\n    await page.getByPlaceholder('What needs to be done?').fill('Buy groceries');\n    await page.getByRole('button', { name: 'Add' }).click();\n\n    await expect(page.getByText('Buy groceries')).toBeVisible();\n    await expect(page.getByTestId('todo-count')).toHaveText('1 item');\n  });\n\n  test('completes a todo', async ({ page }) => {\n    // Add a todo first\n    await page.getByPlaceholder('What needs to be done?').fill('Test task');\n    await page.getByRole('button', { name: 'Add' }).click();\n\n    // Complete it\n    await page.getByRole('checkbox', { name: 'Test task' }).check();\n\n    await expect(page.getByText('Test task')).toHaveCSS(\n      'text-decoration-line', 'line-through'\n    );\n  });\n\n  test('persists todos across page reload', async ({ page }) => {\n    await page.getByPlaceholder('What needs to be done?').fill('Persistent task');\n    await page.getByRole('button', { name: 'Add' }).click();\n\n    await page.reload();\n\n    await expect(page.getByText('Persistent task')).toBeVisible();\n  });\n});\n```\n\n**Run:**\n```bash\nnpx playwright test\nnpx playwright test --ui     # interactive mode\nnpx playwright show-report    # view HTML report\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 7. Performance
# ---------------------------------------------------------------------------

def gen_performance() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "How do you implement code splitting and lazy loading in React?",
        "**React.lazy + Suspense** for route-based code splitting:\n```tsx\nimport { lazy, Suspense } from 'react';\nimport { BrowserRouter, Routes, Route } from 'react-router-dom';\n\n// Lazy-loaded routes — each becomes a separate chunk\nconst Home = lazy(() => import('./pages/Home'));\nconst Dashboard = lazy(() => import('./pages/Dashboard'));\nconst Settings = lazy(() => import('./pages/Settings'));\n\nfunction App() {\n  return (\n    <BrowserRouter>\n      <Suspense fallback={<div className=\"loading-spinner\" />}>\n        <Routes>\n          <Route path=\"/\" element={<Home />} />\n          <Route path=\"/dashboard\" element={<Dashboard />} />\n          <Route path=\"/settings\" element={<Settings />} />\n        </Routes>\n      </Suspense>\n    </BrowserRouter>\n  );\n}\n```\n\n**Component-level lazy loading:**\n```tsx\nconst HeavyChart = lazy(() => import('./components/HeavyChart'));\n\nfunction Analytics() {\n  const [showChart, setShowChart] = useState(false);\n\n  return (\n    <div>\n      <button onClick={() => setShowChart(true)}>Show Chart</button>\n      {showChart && (\n        <Suspense fallback={<Skeleton height={400} />}>\n          <HeavyChart />\n        </Suspense>\n      )}\n    </div>\n  );\n}\n```\n\n**Preloading on hover/intent:**\n```tsx\nconst Dashboard = lazy(() => import('./pages/Dashboard'));\n\n// Preload when user hovers over the link\nfunction NavLink() {\n  const preload = () => import('./pages/Dashboard');\n\n  return (\n    <Link\n      to=\"/dashboard\"\n      onMouseEnter={preload}\n      onFocus={preload}\n    >\n      Dashboard\n    </Link>\n  );\n}\n```\n\n**Named exports with lazy:**\n```tsx\n// lazy only works with default exports, so wrap:\nconst MyComponent = lazy(() =>\n  import('./components').then(mod => ({ default: mod.MyComponent }))\n);\n```"
    ))

    pairs.append(msg(
        "How do you optimize React rendering performance?",
        "**1. React.memo — prevent re-renders when props haven't changed:**\n```tsx\nconst ExpensiveList = memo(function ExpensiveList({ items }: { items: Item[] }) {\n  return items.map(item => <ExpensiveItem key={item.id} item={item} />);\n});\n\n// With custom comparison\nconst UserCard = memo(\n  ({ user }: { user: User }) => <div>{user.name}</div>,\n  (prevProps, nextProps) => prevProps.user.id === nextProps.user.id\n);\n```\n\n**2. useMemo — memoize expensive computations:**\n```tsx\nfunction SearchResults({ items, query }: Props) {\n  const filtered = useMemo(\n    () => items.filter(i => i.name.toLowerCase().includes(query.toLowerCase())),\n    [items, query]  // recompute only when these change\n  );\n  return <List items={filtered} />;\n}\n```\n\n**3. useCallback — stable function references:**\n```tsx\nfunction Parent() {\n  const [count, setCount] = useState(0);\n\n  // Without useCallback: new function every render → Child re-renders\n  const handleClick = useCallback((id: string) => {\n    console.log('clicked', id);\n  }, []); // stable reference\n\n  return (\n    <>\n      <button onClick={() => setCount(c => c + 1)}>{count}</button>\n      <MemoizedChild onClick={handleClick} /> {/* Won't re-render */}\n    </>\n  );\n}\n```\n\n**4. Virtualization for long lists:**\n```tsx\nimport { useVirtualizer } from '@tanstack/react-virtual';\n\nfunction VirtualList({ items }: { items: Item[] }) {\n  const parentRef = useRef<HTMLDivElement>(null);\n  const virtualizer = useVirtualizer({\n    count: items.length,\n    getScrollElement: () => parentRef.current,\n    estimateSize: () => 50, // estimated row height\n  });\n\n  return (\n    <div ref={parentRef} style={{ height: 400, overflow: 'auto' }}>\n      <div style={{ height: virtualizer.getTotalSize() }}>\n        {virtualizer.getVirtualItems().map(row => (\n          <div key={row.key}\n               style={{ height: row.size, transform: `translateY(${row.start}px)` }}>\n            {items[row.index].name}\n          </div>\n        ))}\n      </div>\n    </div>\n  );\n}\n```\n\n**5. Key prop optimization:**\n```tsx\n// BAD: index as key causes re-mount on reorder\n{items.map((item, i) => <Item key={i} />)}\n\n// GOOD: stable unique key preserves state\n{items.map(item => <Item key={item.id} />)}\n```"
    ))

    return pairs


# ---------------------------------------------------------------------------
# 8. Accessibility
# ---------------------------------------------------------------------------

def gen_accessibility() -> list[dict]:
    pairs = []

    pairs.append(msg(
        "What are the essential accessibility practices for React applications?",
        "**1. Semantic HTML:**\n```tsx\n// BAD\n<div onClick={handleClick}>Click me</div>\n<div className=\"header\">Title</div>\n\n// GOOD\n<button onClick={handleClick}>Click me</button>\n<h1>Title</h1>\n<nav aria-label=\"Main navigation\">\n  <ul>{/* links */}</ul>\n</nav>\n<main>Content</main>\n<aside>Sidebar</aside>\n```\n\n**2. ARIA attributes:**\n```tsx\n// Modal\n<div role=\"dialog\" aria-modal=\"true\" aria-labelledby=\"modal-title\">\n  <h2 id=\"modal-title\">Confirm Action</h2>\n  <p>Are you sure?</p>\n  <button onClick={onConfirm}>Yes</button>\n  <button onClick={onCancel}>Cancel</button>\n</div>\n\n// Loading state\n<div aria-live=\"polite\" aria-busy={isLoading}>\n  {isLoading ? <Spinner /> : <Results />}\n</div>\n\n// Custom toggle\n<button\n  role=\"switch\"\n  aria-checked={isEnabled}\n  onClick={() => setEnabled(!isEnabled)}\n>\n  Dark Mode: {isEnabled ? 'On' : 'Off'}\n</button>\n\n// Form errors\n<input\n  aria-invalid={!!error}\n  aria-describedby={error ? 'email-error' : undefined}\n/>\n{error && <span id=\"email-error\" role=\"alert\">{error}</span>}\n```\n\n**3. Keyboard navigation:**\n```tsx\nfunction Menu({ items }: { items: MenuItem[] }) {\n  const [focusIndex, setFocusIndex] = useState(0);\n\n  const handleKeyDown = (e: React.KeyboardEvent) => {\n    switch (e.key) {\n      case 'ArrowDown':\n        e.preventDefault();\n        setFocusIndex(i => Math.min(i + 1, items.length - 1));\n        break;\n      case 'ArrowUp':\n        e.preventDefault();\n        setFocusIndex(i => Math.max(i - 1, 0));\n        break;\n      case 'Home':\n        setFocusIndex(0);\n        break;\n      case 'End':\n        setFocusIndex(items.length - 1);\n        break;\n    }\n  };\n\n  return (\n    <ul role=\"menu\" onKeyDown={handleKeyDown}>\n      {items.map((item, i) => (\n        <li key={item.id} role=\"menuitem\" tabIndex={i === focusIndex ? 0 : -1}\n            ref={el => { if (i === focusIndex) el?.focus(); }}>\n          {item.label}\n        </li>\n      ))}\n    </ul>\n  );\n}\n```\n\n**4. Focus management:**\n```tsx\n// Trap focus in modal\nfunction Modal({ isOpen, onClose, children }) {\n  const firstFocusRef = useRef<HTMLButtonElement>(null);\n\n  useEffect(() => {\n    if (isOpen) firstFocusRef.current?.focus();\n  }, [isOpen]);\n\n  useEffect(() => {\n    const handleEscape = (e: KeyboardEvent) => {\n      if (e.key === 'Escape') onClose();\n    };\n    document.addEventListener('keydown', handleEscape);\n    return () => document.removeEventListener('keydown', handleEscape);\n  }, [onClose]);\n\n  if (!isOpen) return null;\n  return (\n    <div role=\"dialog\" aria-modal=\"true\">\n      <button ref={firstFocusRef} onClick={onClose}>Close</button>\n      {children}\n    </div>\n  );\n}\n```\n\n**5. Testing accessibility:**\n```bash\nnpm install -D @axe-core/react\n```\n```tsx\n// In development only\nif (process.env.NODE_ENV === 'development') {\n  import('@axe-core/react').then(axe => {\n    axe.default(React, ReactDOM, 1000);\n  });\n}\n```"
    ))

    pairs.append(msg(
        "How do you build an accessible form in React?",
        "```tsx\nimport { useState } from 'react';\n\ninterface FormErrors {\n  name?: string;\n  email?: string;\n  password?: string;\n}\n\nfunction SignupForm() {\n  const [errors, setErrors] = useState<FormErrors>({});\n  const [submitted, setSubmitted] = useState(false);\n\n  const validate = (data: FormData): FormErrors => {\n    const errs: FormErrors = {};\n    const name = data.get('name') as string;\n    const email = data.get('email') as string;\n    const password = data.get('password') as string;\n\n    if (!name.trim()) errs.name = 'Name is required';\n    if (!email.includes('@')) errs.email = 'Valid email is required';\n    if (password.length < 8) errs.password = 'Password must be at least 8 characters';\n    return errs;\n  };\n\n  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {\n    e.preventDefault();\n    const data = new FormData(e.currentTarget);\n    const errs = validate(data);\n    setErrors(errs);\n\n    if (Object.keys(errs).length === 0) {\n      setSubmitted(true);\n    }\n  };\n\n  if (submitted) {\n    return (\n      <div role=\"status\" aria-live=\"polite\">\n        <h2>Account created successfully!</h2>\n      </div>\n    );\n  }\n\n  return (\n    <form onSubmit={handleSubmit} noValidate aria-label=\"Sign up form\">\n      {/* Announce errors to screen readers */}\n      {Object.keys(errors).length > 0 && (\n        <div role=\"alert\" aria-live=\"assertive\">\n          <p>Please fix the following errors:</p>\n          <ul>\n            {Object.values(errors).map((err, i) => (\n              <li key={i}>{err}</li>\n            ))}\n          </ul>\n        </div>\n      )}\n\n      <div>\n        <label htmlFor=\"name\">Name *</label>\n        <input\n          id=\"name\"\n          name=\"name\"\n          type=\"text\"\n          required\n          aria-required=\"true\"\n          aria-invalid={!!errors.name}\n          aria-describedby={errors.name ? 'name-error' : undefined}\n          autoComplete=\"name\"\n        />\n        {errors.name && (\n          <span id=\"name-error\" role=\"alert\" className=\"text-red-500\">\n            {errors.name}\n          </span>\n        )}\n      </div>\n\n      <div>\n        <label htmlFor=\"email\">Email *</label>\n        <input\n          id=\"email\"\n          name=\"email\"\n          type=\"email\"\n          required\n          aria-required=\"true\"\n          aria-invalid={!!errors.email}\n          aria-describedby={errors.email ? 'email-error' : 'email-hint'}\n          autoComplete=\"email\"\n        />\n        <span id=\"email-hint\" className=\"text-gray-500\">\n          We'll never share your email\n        </span>\n        {errors.email && (\n          <span id=\"email-error\" role=\"alert\" className=\"text-red-500\">\n            {errors.email}\n          </span>\n        )}\n      </div>\n\n      <div>\n        <label htmlFor=\"password\">Password *</label>\n        <input\n          id=\"password\"\n          name=\"password\"\n          type=\"password\"\n          required\n          aria-required=\"true\"\n          aria-invalid={!!errors.password}\n          aria-describedby=\"password-requirements\"\n          autoComplete=\"new-password\"\n          minLength={8}\n        />\n        <span id=\"password-requirements\" className=\"text-gray-500\">\n          Must be at least 8 characters\n        </span>\n      </div>\n\n      <button type=\"submit\">Create Account</button>\n    </form>\n  );\n}\n```\n\n**Key accessibility features:**\n- `<label htmlFor>` linked to inputs\n- `aria-required`, `aria-invalid`, `aria-describedby`\n- Error announcements via `role=\"alert\"` and `aria-live`\n- `autoComplete` hints for password managers\n- Focus management on error (move focus to first error field)\n- Semantic `<form>` with `aria-label`"
    ))

    return pairs


# ---------------------------------------------------------------------------
# Extra categories for more coverage
# ---------------------------------------------------------------------------

def gen_react_patterns() -> list[dict]:
    """Common React patterns and best practices."""
    pairs = []

    pairs.append(msg(
        "How do you implement a custom hook for form handling in React?",
        """```tsx
import { useState, useCallback } from 'react';

interface UseFormOptions<T> {
  initialValues: T;
  validate?: (values: T) => Partial<Record<keyof T, string>>;
  onSubmit: (values: T) => void | Promise<void>;
}

function useForm<T extends Record<string, any>>({
  initialValues,
  validate,
  onSubmit,
}: UseFormOptions<T>) {
  const [values, setValues] = useState<T>(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof T, string>>>({});
  const [touched, setTouched] = useState<Partial<Record<keyof T, boolean>>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = useCallback((
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    const checked = (e.target as HTMLInputElement).checked;
    setValues(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
  }, []);

  const handleBlur = useCallback((
    e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    if (validate) {
      const errs = validate(values);
      setErrors(errs);
    }
  }, [values, validate]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const errs = validate?.(values) ?? {};
    setErrors(errs);
    setTouched(
      Object.keys(values).reduce((acc, k) => ({ ...acc, [k]: true }), {})
    );

    if (Object.keys(errs).length === 0) {
      setIsSubmitting(true);
      try {
        await onSubmit(values);
      } finally {
        setIsSubmitting(false);
      }
    }
  }, [values, validate, onSubmit]);

  const getFieldProps = (name: keyof T) => ({
    name,
    value: values[name] ?? '',
    onChange: handleChange,
    onBlur: handleBlur,
    'aria-invalid': touched[name] && !!errors[name],
  });

  const reset = () => {
    setValues(initialValues);
    setErrors({});
    setTouched({});
  };

  return {
    values, errors, touched, isSubmitting,
    handleChange, handleBlur, handleSubmit,
    getFieldProps, setValues, reset,
  };
}

// Usage
function LoginForm() {
  const form = useForm({
    initialValues: { email: '', password: '' },
    validate: (v) => {
      const errors: Record<string, string> = {};
      if (!v.email) errors.email = 'Required';
      if (!v.password) errors.password = 'Required';
      return errors;
    },
    onSubmit: async (values) => {
      await login(values.email, values.password);
    },
  });

  return (
    <form onSubmit={form.handleSubmit}>
      <input {...form.getFieldProps('email')} type="email" />
      {form.touched.email && form.errors.email && <span>{form.errors.email}</span>}
      <input {...form.getFieldProps('password')} type="password" />
      <button disabled={form.isSubmitting}>Login</button>
    </form>
  );
}
```"""
    ))

    pairs.append(msg(
        "How do you implement the compound component pattern in React?",
        """The **compound component pattern** creates components that share implicit state, like `<select>` + `<option>`.

```tsx
import { createContext, useContext, useState, ReactNode } from 'react';

// Context for shared state
interface AccordionContext {
  openItems: Set<string>;
  toggle: (id: string) => void;
}

const AccordionCtx = createContext<AccordionContext | null>(null);

function useAccordion() {
  const ctx = useContext(AccordionCtx);
  if (!ctx) throw new Error('Accordion.* must be used within <Accordion>');
  return ctx;
}

// Root component
interface AccordionProps {
  children: ReactNode;
  multiple?: boolean; // allow multiple open items
}

function Accordion({ children, multiple = false }: AccordionProps) {
  const [openItems, setOpenItems] = useState<Set<string>>(new Set());

  const toggle = (id: string) => {
    setOpenItems(prev => {
      const next = new Set(multiple ? prev : []);
      if (prev.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <AccordionCtx.Provider value={{ openItems, toggle }}>
      <div role="region">{children}</div>
    </AccordionCtx.Provider>
  );
}

// Item component
function Item({ id, children }: { id: string; children: ReactNode }) {
  return <div data-accordion-item={id}>{children}</div>;
}

// Trigger component
function Trigger({ id, children }: { id: string; children: ReactNode }) {
  const { openItems, toggle } = useAccordion();
  const isOpen = openItems.has(id);

  return (
    <button
      onClick={() => toggle(id)}
      aria-expanded={isOpen}
      aria-controls={`accordion-panel-${id}`}
    >
      {children}
      <span>{isOpen ? '−' : '+'}</span>
    </button>
  );
}

// Panel component
function Panel({ id, children }: { id: string; children: ReactNode }) {
  const { openItems } = useAccordion();
  if (!openItems.has(id)) return null;

  return (
    <div id={`accordion-panel-${id}`} role="region">
      {children}
    </div>
  );
}

// Attach sub-components
Accordion.Item = Item;
Accordion.Trigger = Trigger;
Accordion.Panel = Panel;

// Usage — clean, declarative API
<Accordion multiple>
  <Accordion.Item id="faq-1">
    <Accordion.Trigger id="faq-1">What is React?</Accordion.Trigger>
    <Accordion.Panel id="faq-1">
      React is a JavaScript library for building user interfaces.
    </Accordion.Panel>
  </Accordion.Item>
  <Accordion.Item id="faq-2">
    <Accordion.Trigger id="faq-2">What is JSX?</Accordion.Trigger>
    <Accordion.Panel id="faq-2">
      JSX is a syntax extension that lets you write HTML-like code in JavaScript.
    </Accordion.Panel>
  </Accordion.Item>
</Accordion>
```

This pattern is used by Radix UI, Headless UI, and Reach UI for unstyled, accessible component libraries."""
    ))

    pairs.append(msg(
        "How do you implement error boundaries in React?",
        """Error boundaries catch JavaScript errors in the component tree and display fallback UI.

```tsx
import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log to error reporting service
    this.props.onError?.(error, errorInfo);
    console.error('Error boundary caught:', error, errorInfo);
  }

  reset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError && this.state.error) {
      if (typeof this.props.fallback === 'function') {
        return this.props.fallback(this.state.error, this.reset);
      }
      return this.props.fallback ?? (
        <div role="alert">
          <h2>Something went wrong</h2>
          <button onClick={this.reset}>Try again</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Usage
<ErrorBoundary
  fallback={(error, reset) => (
    <div>
      <p>Error: {error.message}</p>
      <button onClick={reset}>Retry</button>
    </div>
  )}
  onError={(error) => reportToSentry(error)}
>
  <App />
</ErrorBoundary>

// Granular error boundaries
function Dashboard() {
  return (
    <div className="grid grid-cols-3 gap-4">
      <ErrorBoundary fallback={<WidgetError />}>
        <SalesChart />
      </ErrorBoundary>
      <ErrorBoundary fallback={<WidgetError />}>
        <UserStats />
      </ErrorBoundary>
      <ErrorBoundary fallback={<WidgetError />}>
        <RecentActivity />
      </ErrorBoundary>
    </div>
  );
}
```

**Note:** Error boundaries only catch errors during rendering, lifecycle methods, and constructors. They do NOT catch errors in event handlers, async code, or SSR. For those, use try/catch."""
    ))

    pairs.append(msg(
        "How do you implement infinite scrolling in React?",
        """Use **Intersection Observer** to detect when a sentinel element becomes visible.

```tsx
import { useRef, useEffect, useCallback, useState } from 'react';

interface UseInfiniteScrollOptions<T> {
  fetchFn: (page: number) => Promise<{ data: T[]; hasMore: boolean }>;
  initialPage?: number;
}

function useInfiniteScroll<T>({ fetchFn, initialPage = 1 }: UseInfiniteScrollOptions<T>) {
  const [items, setItems] = useState<T[]>([]);
  const [page, setPage] = useState(initialPage);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const observerRef = useRef<IntersectionObserver | null>(null);

  const loadMore = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const result = await fetchFn(page);
      setItems(prev => [...prev, ...result.data]);
      setHasMore(result.hasMore);
      setPage(p => p + 1);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, [page, loading, hasMore, fetchFn]);

  // Ref callback for the sentinel element
  const sentinelRef = useCallback(
    (node: HTMLElement | null) => {
      if (observerRef.current) observerRef.current.disconnect();
      if (!node) return;

      observerRef.current = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting && hasMore && !loading) {
            loadMore();
          }
        },
        { rootMargin: '200px' } // Load before reaching bottom
      );
      observerRef.current.observe(node);
    },
    [hasMore, loading, loadMore]
  );

  // Initial load
  useEffect(() => { loadMore(); }, []); // eslint-disable-line

  return { items, loading, error, hasMore, sentinelRef };
}

// Usage
function PostFeed() {
  const { items, loading, hasMore, sentinelRef } = useInfiniteScroll<Post>({
    fetchFn: async (page) => {
      const res = await fetch(`/api/posts?page=${page}&limit=20`);
      const data = await res.json();
      return { data: data.posts, hasMore: data.hasMore };
    },
  });

  return (
    <div>
      {items.map(post => (
        <PostCard key={post.id} post={post} />
      ))}
      {loading && <Spinner />}
      {hasMore && <div ref={sentinelRef} style={{ height: 1 }} />}
      {!hasMore && <p>No more posts</p>}
    </div>
  );
}
```

**Performance tip:** Combine with virtualization (TanStack Virtual) for very long lists — infinite scroll loads data, virtualization only renders visible rows."""
    ))

    return pairs


def gen_css_patterns() -> list[dict]:
    """CSS layout and styling patterns."""
    pairs = []

    pairs.append(msg(
        "How do you create common CSS Grid layouts with Tailwind?",
        """```tsx
// Dashboard layout: sidebar + header + main
<div className=\"grid grid-cols-[250px_1fr] grid-rows-[64px_1fr] h-screen\">
  <header className=\"col-span-2 bg-white border-b flex items-center px-6\">\n    Header\n  </header>\n  <aside className=\"bg-gray-50 border-r p-4 overflow-y-auto\">\n    Sidebar\n  </aside>\n  <main className=\"p-6 overflow-y-auto\">\n    Content\n  </main>\n</div>

// Responsive card grid with auto-fill
<div className=\"grid grid-cols-[repeat(auto-fill,minmax(280px,1fr))] gap-6\">\n  {items.map(item => <Card key={item.id} item={item} />)}\n</div>

// Holy grail layout
<div className=\"grid grid-cols-[200px_1fr_200px] gap-4 min-h-screen\">\n  <aside>Left sidebar</aside>\n  <main>Main content</main>\n  <aside>Right sidebar</aside>\n</div>

// Masonry-like with CSS columns (not true masonry)
<div className=\"columns-1 sm:columns-2 lg:columns-3 gap-6 space-y-6\">\n  {items.map(item => (\n    <div key={item.id} className=\"break-inside-avoid\">\n      <Card item={item} />\n    </div>\n  ))}\n</div>

// Centered content with max-width
<div className=\"grid place-items-center min-h-screen\">\n  <div className=\"w-full max-w-md p-8\">\n    <LoginForm />\n  </div>\n</div>

// Sticky footer
<div className=\"grid grid-rows-[auto_1fr_auto] min-h-screen\">\n  <header>Header</header>\n  <main>Content (grows to fill space)</main>\n  <footer>Always at bottom</footer>\n</div>
```"""
    ))

    pairs.append(msg(
        "How do you create smooth CSS animations and transitions with Tailwind?",
        """```tsx
// Hover effects
<button className=\"\n  transform transition-all duration-200 ease-out\n  hover:scale-105 hover:shadow-lg\n  active:scale-95\n\">\n  Hover me\n</button>

// Fade in on mount (using CSS animation)
<div className=\"animate-in fade-in duration-500\">\n  Content\n</div>

// Slide in from bottom
<div className=\"animate-in slide-in-from-bottom-4 duration-300\">\n  Content\n</div>

// Custom keyframe animation in Tailwind v4:
/* In CSS */
@keyframes float {\n  0%, 100% { transform: translateY(0); }\n  50% { transform: translateY(-10px); }\n}

@utility animate-float {\n  animation: float 3s ease-in-out infinite;\n}

/* In component */
<div className=\"animate-float\">Floating element</div>

// Transition on state change
function Dropdown({ isOpen }: { isOpen: boolean }) {\n  return (\n    <div className={`\n      transition-all duration-200 ease-out\n      ${isOpen\n        ? 'opacity-100 translate-y-0 pointer-events-auto'\n        : 'opacity-0 -translate-y-2 pointer-events-none'\n      }\n    `}>\n      Dropdown content\n    </div>\n  );\n}

// Skeleton loading animation
<div className=\"animate-pulse space-y-4\">\n  <div className=\"h-4 bg-gray-200 rounded w-3/4\" />\n  <div className=\"h-4 bg-gray-200 rounded w-1/2\" />\n  <div className=\"h-32 bg-gray-200 rounded\" />\n</div>

// Staggered animation with delay
{items.map((item, i) => (\n  <div\n    key={item.id}\n    className=\"animate-in fade-in slide-in-from-bottom-2\"\n    style={{ animationDelay: `${i * 100}ms`, animationFillMode: 'backwards' }}\n  >\n    {item.name}\n  </div>\n))}
```"""
    ))

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    generators = {
        "React 19": gen_react19,
        "Vite Configuration": gen_vite,
        "Tailwind CSS": gen_tailwind,
        "TypeScript Frontend": gen_typescript_frontend,
        "State Management": gen_state_management,
        "Testing": gen_testing,
        "Performance": gen_performance,
        "Accessibility": gen_accessibility,
        "React Patterns": gen_react_patterns,
        "CSS Patterns": gen_css_patterns,
    }

    total = 0
    for label, gen_fn in generators.items():
        pairs = gen_fn()
        count = len(pairs)
        total += count
        logger.info("  %-25s %5d pairs", label, count)
        for pair in pairs:
            print(json.dumps(pair, ensure_ascii=False))

    logger.info("=== TOTAL: %d pairs ===", total)


if __name__ == "__main__":
    main()
