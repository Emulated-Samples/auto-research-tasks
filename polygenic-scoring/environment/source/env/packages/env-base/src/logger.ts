/**
 * Logger implementations
 */

import type { Logger } from "./types.js";

/**
 * Console logger - outputs to stdout
 */
export class ConsoleLogger implements Logger {
  info(message: string): void {
    console.log(message);
  }

  error(message: string): void {
    console.error(message);
  }

  warn(message: string): void {
    console.warn(message);
  }

  debug(message: string): void {
    console.debug(message);
  }
}

/**
 * Silent logger - suppresses all output
 */
export class SilentLogger implements Logger {
  info(_message: string): void {}
  error(_message: string): void {}
  warn(_message: string): void {}
  debug(_message: string): void {}
}
