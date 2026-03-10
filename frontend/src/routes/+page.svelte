<script>
  import { sendQuery, getToken } from '$lib/api';
  import { NAMESPACES } from '$lib/ai';

  let messages = $state([]);
  let input = $state('');
  let loading = $state(false);
  let namespace = $state('customer_support');
  let conversationId = $state(null);
  let isLoggedIn = $state(false);

  $effect(() => {
    isLoggedIn = !!getToken();
  });

  async function send() {
    if (!input.trim() || loading) return;
    const query = input.trim();
    input = '';
    messages = [...messages, { role: 'user', content: query }];
    loading = true;

    try {
      const res = await sendQuery(query, namespace, conversationId);
      conversationId = res.conversation_id;
      messages = [...messages, {
        role: 'assistant',
        content: res.answer,
        latency: res.latency_ms,
        sources: res.sources,
      }];
    } catch (e) {
      messages = [...messages, { role: 'assistant', content: `Error: ${e.message}` }];
    } finally {
      loading = false;
    }
  }

  function handleKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function newChat() {
    messages = [];
    conversationId = null;
  }
</script>

{#if !isLoggedIn}
  <div class="min-h-screen flex items-center justify-center">
    <div class="text-center">
      <h1 class="text-3xl font-bold mb-4">AlvinAI</h1>
      <p class="text-gray-400 mb-6">Automotive AI Assistant</p>
      <a href="/login" class="bg-alvin-600 hover:bg-alvin-700 px-6 py-3 rounded-lg text-white">
        Sign In
      </a>
    </div>
  </div>
{:else}
  <div class="flex h-screen">
    <!-- Sidebar -->
    <div class="w-64 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div class="p-4 border-b border-gray-800">
        <h1 class="text-xl font-bold text-alvin-500">AlvinAI</h1>
        <p class="text-xs text-gray-500 mt-1">Automotive AI Assistant</p>
      </div>
      <div class="p-3">
        <button onclick={newChat} class="w-full bg-alvin-600 hover:bg-alvin-700 rounded-lg px-4 py-2 text-sm">
          + New Chat
        </button>
      </div>
      <div class="p-3">
        <label class="text-xs text-gray-500 uppercase tracking-wider">Namespace</label>
        <select bind:value={namespace} class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm">
          {#each NAMESPACES as ns}
            <option value={ns.id}>{ns.icon} {ns.label}</option>
          {/each}
        </select>
      </div>
      <div class="flex-1"></div>
      <div class="p-3 border-t border-gray-800">
        <button onclick={() => { import('$lib/api').then(m => { m.clearToken(); isLoggedIn = false; }) }}
          class="text-xs text-gray-500 hover:text-gray-300">Sign Out</button>
      </div>
    </div>

    <!-- Chat area -->
    <div class="flex-1 flex flex-col">
      <!-- Messages -->
      <div class="flex-1 overflow-y-auto p-6 space-y-4">
        {#if messages.length === 0}
          <div class="flex items-center justify-center h-full">
            <div class="text-center text-gray-500">
              <p class="text-2xl mb-2">Ask AlvinAI anything</p>
              <p class="text-sm">Select a namespace and start asking questions</p>
            </div>
          </div>
        {/if}
        {#each messages as msg}
          <div class="flex {msg.role === 'user' ? 'justify-end' : 'justify-start'}">
            <div class="max-w-2xl px-4 py-3 rounded-2xl {msg.role === 'user' ? 'bg-alvin-600 text-white' : 'bg-gray-800 text-gray-100'}">
              <p class="whitespace-pre-wrap">{msg.content}</p>
              {#if msg.latency}
                <p class="text-xs mt-2 opacity-50">{msg.latency.toFixed(0)}ms</p>
              {/if}
            </div>
          </div>
        {/each}
        {#if loading}
          <div class="flex justify-start">
            <div class="bg-gray-800 px-4 py-3 rounded-2xl">
              <div class="flex space-x-1">
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Input -->
      <div class="border-t border-gray-800 p-4">
        <div class="flex gap-3 max-w-4xl mx-auto">
          <textarea
            bind:value={input}
            onkeydown={handleKeydown}
            placeholder="Ask a question..."
            rows="1"
            class="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 resize-none focus:outline-none focus:border-alvin-500"
          ></textarea>
          <button onclick={send} disabled={loading || !input.trim()}
            class="bg-alvin-600 hover:bg-alvin-700 disabled:opacity-50 rounded-xl px-6 py-3 font-medium">
            Send
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}
