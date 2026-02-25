import { useEffect, useRef } from 'react';
import { registerEscapeClose } from '../utils/escapeModalStack';

export function useEscapeClose(active: boolean, onClose: () => void) {
  const closeRef = useRef(onClose);

  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!active) return;
    const unregister = registerEscapeClose(() => closeRef.current());
    return () => unregister();
  }, [active]);
}
