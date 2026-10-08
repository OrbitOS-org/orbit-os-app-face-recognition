package main

import (
	"net"
	"strings"
	"testing"
)

// Two copies of the app on one device must not end up on the same port: the
// second one takes the next port of the reserved range.
func TestPageTakesTheNextFreePort(t *testing.T) {
	first, _, err := listenAndRegister(&FaceClient{cfg: hardcodedConfig()})
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, _, err := listenAndRegister(&FaceClient{cfg: hardcodedConfig()})
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()

	for _, l := range []net.Listener{first, second} {
		addr := l.Addr().(*net.TCPAddr)
		if !addr.IP.IsLoopback() || addr.Port < portMin || addr.Port > portMax {
			t.Fatalf("listening on %v, want 127.0.0.1 and a port in %d-%d", addr, portMin, portMax)
		}
	}
	if first.Addr().String() == second.Addr().String() {
		t.Fatalf("both on %v", first.Addr())
	}
}

// With a fixed address (development), a busy port is an error, not a silent start without a page.
func TestFixedAddressBusyIsAnError(t *testing.T) {
	busy, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer busy.Close()

	cfg := hardcodedConfig()
	cfg.WebUIListen = busy.Addr().String()
	if _, _, err := listenAndRegister(&FaceClient{cfg: cfg}); err == nil || !strings.Contains(err.Error(), "listen on") {
		t.Fatalf("busy port: %v", err)
	}
	c := &FaceClient{cfg: cfg}
	if err := startWebUI(c); err == nil {
		t.Fatal("the page started on a busy port")
	}
}
