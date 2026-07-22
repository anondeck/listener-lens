import { Container } from '@cloudflare/containers';

const MAX_RENDER_CALLS = 5;

export class TransformContainer extends Container {
  defaultPort = 8080;
  sleepAfter = '2m';
  enableInternet = false;

  async reserveRenders(clientHash, requested, day) {
    if (!/^[a-f0-9]{64}$/.test(clientHash) || !Number.isInteger(requested) || requested < 1 || requested > MAX_RENDER_CALLS || !/^\d{4}-\d{2}-\d{2}$/.test(day)) {
      return { allowed: false, reason: 'invalid_budget_request' };
    }
    return this.ctx.storage.transaction(async transaction => {
      const globalKey = `budget:${day}:global`;
      const clientKey = `budget:${day}:client:${clientHash}`;
      const values = await transaction.get([globalKey, clientKey]);
      const globalUsed = Number(values.get(globalKey) || 0);
      const clientUsed = Number(values.get(clientKey) || 0);
      if (globalUsed + requested > 100) return { allowed: false, reason: 'daily_global_budget' };
      if (clientUsed + requested > 5) return { allowed: false, reason: 'daily_client_budget' };
      await transaction.put({ [globalKey]: globalUsed + requested, [clientKey]: clientUsed + requested });
      return { allowed: true, reserved: requested, global_used: globalUsed + requested, client_used: clientUsed + requested };
    });
  }

  async releaseRenders(clientHash, released, day) {
    if (!/^[a-f0-9]{64}$/.test(clientHash) || !Number.isInteger(released) || released < 0 || released > MAX_RENDER_CALLS || !/^\d{4}-\d{2}-\d{2}$/.test(day)) return;
    await this.ctx.storage.transaction(async transaction => {
      const globalKey = `budget:${day}:global`;
      const clientKey = `budget:${day}:client:${clientHash}`;
      const values = await transaction.get([globalKey, clientKey]);
      await transaction.put({
        [globalKey]: Math.max(0, Number(values.get(globalKey) || 0) - released),
        [clientKey]: Math.max(0, Number(values.get(clientKey) || 0) - released)
      });
    });
  }
}
