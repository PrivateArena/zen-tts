package main

import (
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
)

// --- THEME ---
var (
	ColorBg      = tcell.NewHexColor(0x121212)
	ColorPanel   = tcell.NewHexColor(0x282828)
	ColorBorder  = tcell.NewHexColor(0x444444)
	ColorFocus   = tcell.NewHexColor(0x5f87af)
	ColorText    = tcell.NewHexColor(0xeeeeee)
	ColorDim     = tcell.NewHexColor(0xaaaaaa)
	ColorSuccess = tcell.ColorGreen
	ColorWarn    = tcell.ColorYellow
)

func setupTheme() {
	tview.Styles.PrimitiveBackgroundColor = ColorBg
	tview.Styles.ContrastBackgroundColor = ColorPanel
	tview.Styles.MoreContrastBackgroundColor = ColorPanel
	tview.Styles.BorderColor = ColorBorder
	tview.Styles.TitleColor = ColorFocus
	tview.Styles.PrimaryTextColor = ColorText
	tview.Styles.SecondaryTextColor = ColorDim
	tview.Styles.TertiaryTextColor = ColorSuccess
}

var (
	app       *tview.Application
	serverLog *tview.TextView
)

// --- LOGGING (Thread-Safe) ---
func LogMsg(msg string) {
	// If UI is running, send to UI thread queue
	if app != nil && serverLog != nil {
		app.QueueUpdateDraw(func() {
			fmt.Fprintf(serverLog, "[%s] %s\n", time.Now().Format("15:04:05"), msg)
			serverLog.ScrollToEnd()
		})
	} else {
		// Headless mode fallback
		log.Println(msg)
	}
}

// --- SCREENS ---

func buildDashboard() *tview.Flex {
	serverLog = tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetMaxLines(1000)
	serverLog.SetBorder(true).SetTitle(" Engine Logs ")
	serverLog.SetBackgroundColor(ColorBg)

	infoPanel := tview.NewTextView().SetDynamicColors(true)
	infoPanel.SetBorder(true).SetTitle(" Status ")
	
	// Background Status Updater
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		for range ticker.C {
			if app != nil {
				app.QueueUpdateDraw(func() {
					status := "[red]STOPPED[-]"
					if ServerActive {
						status = fmt.Sprintf("[green]RUNNING (:%d)[-]", ServerPort)
					}
					txt := fmt.Sprintf("\n  Server: %s\n  Voice:  [#5f87af]%s[-]\n  Port:   %d\n\n  [gray]Use Sidebar to Control[-]", 
						status, CurrentConfig.LastModel, CurrentConfig.Port)
					infoPanel.SetText(txt)
				})
			}
		}
	}()

	return tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(infoPanel, 8, 1, false).
		AddItem(serverLog, 0, 1, false)
}

func buildVoices(pages *tview.Pages, actions *tview.List) *tview.Flex {
	table := tview.NewTable().SetBorders(false).SetSelectable(true, false)
	table.SetFixed(1, 0).SetBackgroundColor(ColorBg)

	search := tview.NewInputField().SetLabel("  Search: ").SetLabelColor(ColorFocus)
	search.SetFieldBackgroundColor(ColorPanel).SetFieldTextColor(ColorText)

	// Sort Data
	type row struct { key, lang, qual string }
	var rows []row
	for k, v := range Registry {
		rows = append(rows, row{k, v.Language.NameEnglish, v.Quality})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].lang < rows[j].lang })

	filtered := rows

	refresh := func() {
		table.Clear()
		headers := []string{"Language", "Voice Key", "Quality"}
		for i, h := range headers {
			table.SetCell(0, i, tview.NewTableCell(fmt.Sprintf(" %s ", h)).SetTextColor(ColorFocus).SetSelectable(false))
		}
		
		for i, r := range filtered {
			y := i + 1
			cQual := ColorDim
			if r.qual == "high" { cQual = ColorSuccess }
			if r.qual == "medium" { cQual = ColorWarn }
			
			prefix := " "
			if r.key == CurrentConfig.LastModel { prefix = " ★ " }

			table.SetCell(y, 0, tview.NewTableCell(prefix + r.lang).SetTextColor(ColorText))
			table.SetCell(y, 1, tview.NewTableCell(r.key).SetTextColor(ColorFocus))
			table.SetCell(y, 2, tview.NewTableCell(r.qual).SetTextColor(cQual))
		}
		table.ScrollToBeginning()
	}

	search.SetChangedFunc(func(text string) {
		term := strings.ToLower(text)
		filtered = nil
		for _, r := range rows {
			if strings.Contains(strings.ToLower(r.key), term) || strings.Contains(strings.ToLower(r.lang), term) {
				filtered = append(filtered, r)
			}
		}
		refresh()
	})

	table.SetSelectedFunc(func(row, col int) {
		if row > 0 && row <= len(filtered) {
			sel := filtered[row-1]
			CurrentConfig.LastModel = sel.key
			SaveConfig()
			
			// FIX: Run server operations in background goroutine to avoid UI freeze
			go func() {
				LogMsg(fmt.Sprintf("[yellow]Switching to %s...[-]", sel.key))
				if ServerActive {
					StopServer()
					time.Sleep(200 * time.Millisecond) // Safe to sleep in goroutine
				}
				StartServer(sel.key, CurrentConfig.Port, 0)
			}()
			
			pages.SwitchToPage("Dashboard")
			actions.SetCurrentItem(0)
			app.SetFocus(actions)
		}
	})

	refresh()

	return tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(search, 1, 0, false).
		AddItem(tview.NewBox(), 1, 0, false).
		AddItem(table, 0, 1, true)
}

func buildRules() *tview.Flex {
	table := tview.NewTable().SetBorders(false).SetSelectable(true, false)
	table.SetBackgroundColor(ColorBg)
	
	refresh := func() {
		table.Clear()
		table.SetCell(0, 0, tview.NewTableCell(" Type ").SetTextColor(ColorFocus).SetSelectable(false))
		table.SetCell(0, 1, tview.NewTableCell(" Pattern ").SetTextColor(ColorFocus).SetSelectable(false))
		table.SetCell(0, 2, tview.NewTableCell(" Replacement ").SetTextColor(ColorFocus).SetSelectable(false))

		for i, rule := range CurrentConfig.Replacements {
			y := i + 1
			typeStr := "TEXT"
			if rule.IsRegex { typeStr = "REGEX" }
			table.SetCell(y, 0, tview.NewTableCell(typeStr).SetTextColor(ColorDim))
			table.SetCell(y, 1, tview.NewTableCell(rule.Pattern).SetTextColor(ColorText))
			table.SetCell(y, 2, tview.NewTableCell(rule.Replacement).SetTextColor(ColorSuccess))
		}
	}

	help := tview.NewTextView().SetDynamicColors(true).SetTextAlign(tview.AlignCenter)
	help.SetText("[gray]Edit 'config.json' manually to add rules (UI Editor coming soon)[-]")

	refresh()
	
	return tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(table, 0, 1, true).
		AddItem(help, 1, 0, false)
}

// --- MAIN RUNNER ---

func RunTUI(port int, cpuCore int) {
	setupTheme()
	app = tview.NewApplication()
	pages := tview.NewPages()

	// Sidebar Navigation
	nav := tview.NewList().ShowSecondaryText(false)
	nav.SetBorder(true).SetTitle(" Navigation ")
	nav.SetSelectedBackgroundColor(ColorFocus).SetMainTextColor(ColorText)
	
	nav.AddItem("Dashboard", "", 'd', func() { pages.SwitchToPage("Dashboard") })
	nav.AddItem("Voices", "", 'v', func() { pages.SwitchToPage("Voices") })
	nav.AddItem("Rules", "", 'r', func() { pages.SwitchToPage("Rules") })
	nav.AddItem("Quit", "", 'q', func() { app.Stop() })

	// Action Bar
	actions := tview.NewList().ShowSecondaryText(false)
	actions.SetBorder(true).SetTitle(" Actions ")
	actions.SetSelectedBackgroundColor(ColorFocus)
	
	actions.AddItem("Start/Stop Server", "", 's', func() { 
		// FIX: Run in background to prevent UI freeze
		go func() {
			ToggleServer(CurrentConfig.LastModel, CurrentConfig.Port, cpuCore) 
		}()
	})

	// Assemble Screens
	pages.AddPage("Dashboard", buildDashboard(), true, true)
	pages.AddPage("Voices", buildVoices(pages, actions), true, false)
	pages.AddPage("Rules", buildRules(), true, false)

	// Layout
	sidebar := tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(nav, 0, 1, true).
		AddItem(actions, 5, 0, false)

	root := tview.NewFlex().
		AddItem(sidebar, 25, 0, true).
		AddItem(pages, 0, 1, false)

	// Global Keybinds
	app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyTab {
			if nav.HasFocus() { app.SetFocus(actions) } else 
			if actions.HasFocus() { app.SetFocus(pages) } else 
			{ app.SetFocus(nav) }
			return nil
		}
		// Ctrl+C Safety Hatch (In case UI logic hangs, this should still catch OS signals if possible, 
		// but tview usually needs to handle it explicitly here)
		if event.Key() == tcell.KeyCtrlC {
			app.Stop()
			return nil
		}
		return event
	})

	if err := app.SetRoot(root, true).EnableMouse(true).Run(); err != nil {
		panic(err)
	}
}