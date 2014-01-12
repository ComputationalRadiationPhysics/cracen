# Speichertransfer zwischen Host und Device

## Grobgliederung
Die Daten werden aus einem Ringpuffer mit den Sampledaten gelesen. 
Um die einzelnen Nodes zu verwalten wird eine Klasse implementiert, deren Objekte jeweils einen der 4 Nodes der GPU verwalten.

## Aufgaben

- Initialisierungscode schreiben
	- Auslesen der Systemdetails mit cudaDeviceProps
	- Objekte für jeden Node erstellen
	- die einzelnen Threads starten

- Node Klasse
	- Attribute
		- Device Pointer
		- Referenz auf Ringpuffer für Eingang
		- Referenz auf Ringpuffer für Ausgang
		- finish 
			- Variable die vom Host gesetzt werden kann um anzuzeigen, dass keine neuen Daten mehr folgen
	- Konstruktor/Destruktor
	- Methoden
		- stop (public)
			- Host beendet den Thread evt. auch im Destruktor implementiert
		- run (public)
			- Wird als eigener Thread gestartet
		- copyToDevice (private)
		- getResults (private)

- Speichertransfer Host -> Device
	- In Methode copyToDevice
	- Textur Referenz anlegen (im File scope)
	- Texturobjekt initialisieren
	- Daten in Textur kopieren
	- Anschließend Kernel starten
	
- Speichertransfer Device -> Host
	- Ende des Kernels abwarten
	- Daten in Ausgangs
	
- Ausgangspuffer
	- Initialisierung
	- Read
	- Write
