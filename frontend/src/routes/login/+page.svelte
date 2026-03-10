<script>
  import { login, register } from '$lib/api';

  let email = $state('');
  let password = $state('');
  let fullName = $state('');
  let isRegister = $state(false);
  let error = $state('');
  let loading = $state(false);

  async function submit() {
    error = '';
    loading = true;
    try {
      if (isRegister) {
        await register(email, password, fullName);
      } else {
        await login(email, password);
      }
      window.location.href = '/';
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }
</script>

<div class="min-h-screen flex items-center justify-center">
  <div class="w-full max-w-sm bg-gray-900 rounded-2xl p-8 border border-gray-800">
    <h1 class="text-2xl font-bold text-center mb-2">AlvinAI</h1>
    <p class="text-gray-500 text-center text-sm mb-6">
      {isRegister ? 'Create an account' : 'Sign in to continue'}
    </p>

    {#if error}
      <div class="bg-red-900/50 border border-red-700 rounded-lg px-4 py-2 mb-4 text-sm text-red-300">
        {error}
      </div>
    {/if}

    <form onsubmit={(e) => { e.preventDefault(); submit(); }}>
      {#if isRegister}
        <input
          bind:value={fullName}
          type="text"
          placeholder="Full name"
          class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 mb-3 focus:outline-none focus:border-alvin-500"
        />
      {/if}
      <input
        bind:value={email}
        type="email"
        placeholder="Email"
        required
        class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 mb-3 focus:outline-none focus:border-alvin-500"
      />
      <input
        bind:value={password}
        type="password"
        placeholder="Password"
        required
        class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 mb-4 focus:outline-none focus:border-alvin-500"
      />
      <button
        type="submit"
        disabled={loading}
        class="w-full bg-alvin-600 hover:bg-alvin-700 disabled:opacity-50 rounded-lg px-4 py-3 font-medium"
      >
        {loading ? '...' : isRegister ? 'Create Account' : 'Sign In'}
      </button>
    </form>

    <p class="text-center text-sm text-gray-500 mt-4">
      <button onclick={() => { isRegister = !isRegister; error = ''; }}
        class="text-alvin-500 hover:underline">
        {isRegister ? 'Already have an account? Sign in' : 'Need an account? Register'}
      </button>
    </p>
  </div>
</div>
