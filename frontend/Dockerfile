FROM node:20-alpine

WORKDIR /app

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy dependency files
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install --frozen-lockfile || pnpm install || true

# Copy application code
COPY . .

RUN pnpm build || true

EXPOSE 3000

CMD ["node", "build"]
